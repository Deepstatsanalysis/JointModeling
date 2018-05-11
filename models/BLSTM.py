from base.base_model import BaseModel

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.layers import flatten

class BLSTM(BaseModel):
    def __init__(self, config, data):
        super(BLSTM, self).__init__(config, data)
        self.build_model()
        self.init_saver()

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def build_model(self):

        with tf.device('/cpu'):
            
            self.weights = self._initialize_weights()
            self.is_training = tf.placeholder(tf.bool)
            self.learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')

            with tf.name_scope('Y'):
                self.y10 = tf.placeholder('float', [self.config.max_length, 10])
                self.y8 = tf.placeholder('float',  [self.config.max_length, 8])
                self.y6 = tf.placeholder('float',  [self.config.max_length, 6])
                self.y4 = tf.placeholder('float',  [self.config.max_length, 4])
                self.y_cont = tf.placeholder('float',  [self.config.max_length, 1])
            with tf.name_scope('X'):
                self.X = tf.placeholder("float", [None, self.config.max_length, self.config.n_input])
                self.x_unstack = tf.unstack(self.X, self.config.max_length, 1)
                self.img_summary_op = tf.reshape(self.X, [-1, self.config.max_length, self.config.n_input, 1])

            with tf.name_scope('blstm'):

                # Forward direction cell
                self.lstm_fw_cells =[]
                for n in range(self.config.n_layers):
                    self.lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(self.config.n_hidden)
                    self.lstm_fw_cells.append(self.lstm_fw_cell)

                # Backward direction cell
                self.lstm_bw_cells =[]
                for n in range(self.config.n_layers):
                    self.lstm_bw_cell = rnn.BasicLSTMCell(self.config.n_hidden)
                    self.lstm_bw_cells.append(self.lstm_bw_cell)

                self.outputs, _, _= tf.contrib.rnn.stack_bidirectional_dynamic_rnn(self.lstm_fw_cells, self.lstm_bw_cells, self.X, dtype=tf.float32)
                self.outputs = tf.concat(self.outputs, 2)[0]


            if self.config.fc_path == 2:
                with tf.name_scope('fc1'):
                    self.fc1 = tf.matmul(self.outputs, self.weights['fc1_w']) + self.weights['fc1_b' ]
                    self.fc1 = tf.sigmoid(self.fc1)
                with tf.name_scope('fc2'):
                    self.fc2 = tf.matmul(self.outputs, self.weights['fc2_w']) + self.weights['fc2_b' ]

                with tf.name_scope('fc1fc2'):
                    self.outputs = tf.transpose(tf.transpose(tf.concat([self.fc1, self.fc2],axis=1)))

            elif self.config.fc_path == 1:
                self.fc = self.outputs
                fc_input = 2*self.config.n_hidden
                for i in range(self.config.fcs_num):
                    with tf.name_scope('fc%d'%(i+1)):
                        self.weights['fc_w%d'%(i+1)] = tf.get_variable("fc_W%d"%(i+1), shape=[fc_input, self.config.n_fc],initializer=tf.contrib.layers.xavier_initializer())
                        self.weights['fc_b%d'%(i+1)] = tf.get_variable("fc_b%d"%(i+1), shape=[self.config.n_fc],initializer=tf.zeros_initializer())
                        self.fc = tf.matmul(self.fc,self.weights['fc_w%d'%(i+1)]) + self.weights['fc_b%d'%(i+1)]
                        
                        if self.config.fc_act == 'sigmoid':
                            self.fc = tf.sigmoid(self.fc)
                        if self.config.fc_act == 'tanh':
                            self.fc = tf.tanh(self.fc)
                        
                        fc_input = self.config.n_fc

                self.outputs = self.fc
                
            with tf.name_scope('Y_pred'):

                self.y_logits10 = [0]
                self.y_logits8  = [0]
                self.y_logits4  = [0]
                self.y_logits6  = [0]


                self.y_logits10= tf.matmul(self.outputs,self.weights['w10']) + self.weights['b10']
                self.y_logits8 = tf.matmul(self.outputs,self.weights['w8']) + self.weights['b8' ]
                self.y_logits6 = tf.matmul(self.outputs,self.weights['w6']) + self.weights['b6' ]
                self.y_logits4 = tf.matmul(self.outputs,self.weights['w4']) + self.weights['b4' ]

                self.y_logits_reg = tf.matmul(self.outputs,self.weights['wr']) + self.weights['br' ]
                
                self.y_pred10 = tf.nn.softmax(self.y_logits10)
                self.y_pred8  = tf.nn.softmax(self.y_logits8 )
                self.y_pred6  = tf.nn.softmax(self.y_logits6 )
                self.y_pred4  = tf.nn.softmax(self.y_logits4 )
                self.y_pred_reg = self.y_logits_reg

                y_pred10_cont = self.get_cont_val(self.y_pred10,10, use_priors=self.config.priors)
                y_pred8_cont  = self.get_cont_val(self.y_pred8 ,8 , use_priors=self.config.priors)
                y_pred6_cont  = self.get_cont_val(self.y_pred6 ,6 , use_priors=self.config.priors)
                y_pred4_cont  = self.get_cont_val(self.y_pred4 ,4 , use_priors=self.config.priors)

                self.yout_inputs = []
                if 'loss10' in self.config['losses']:
                    self.yout_inputs.append(y_pred10_cont)
                if 'loss8' in self.config['losses']:
                    self.yout_inputs.append(y_pred8_cont)
                if 'loss6' in self.config['losses']:
                    self.yout_inputs.append(y_pred6_cont)
                if 'loss4' in self.config['losses']:
                    self.yout_inputs.append(y_pred4_cont)
                if 'rmse' in self.config['losses'] or 'ccc_err' in self.config['losses']:
                    self.yout_inputs.append(tf.reshape(self.y_pred_reg, shape=(7500,)))

                print('Weights = %d'%len(self.yout_inputs))
                self.weights['w'] = tf.get_variable("W", shape=[len(self.yout_inputs), 1],initializer=tf.contrib.layers.xavier_initializer())
                self.yout_inputs = tf.transpose(tf.concat([self.yout_inputs],axis=1))
                self.y_out = tf.transpose(tf.transpose(tf.concat([self.yout_inputs],axis=1)))
                self.y_out = tf.matmul(self.y_out, self.weights['w']) + self.weights['b']


            with tf.name_scope('Loss'):
                if self.config.cost == 'argmax':
                    c10 = tf.cast(1+tf.divide( tf.abs(tf.argmax(self.y_pred10,1)-tf.argmax(self.y10,1)), 10),tf.float32)
                    c8  = tf.cast(1+tf.divide( tf.abs(tf.argmax(self.y_pred8 ,1)-tf.argmax(self.y8 ,1)), 8 ),tf.float32)
                    c6  = tf.cast(1+tf.divide( tf.abs(tf.argmax(self.y_pred6 ,1)-tf.argmax(self.y6 ,1)), 6 ),tf.float32)
                    c4  = tf.cast(1+tf.divide( tf.abs(tf.argmax(self.y_pred4 ,1)-tf.argmax(self.y4 ,1)), 4 ),tf.float32)
                elif self.config.cost == 'norm':
                    c10 = self.norm_cost(self.y10, self.y_pred10,k=10)
                    c8  = self.norm_cost(self.y8 , self.y_pred8 ,k=8 )
                    c6  = self.norm_cost(self.y6 , self.y_pred6 ,k=6 )
                    c4  = self.norm_cost(self.y4 , self.y_pred4 ,k=4 )
                
                self.c_reg =  1 + tf.sqrt((tf.square(self.y_pred_reg - self.y_cont)))
                self.config.l2_beta = 0
                
                if self.config.modified_CE:
                    CE10 = self.modified_CE(logits=self.y_logits10, labels=self.y10, k=10, use_priors=self.config.priors)
                    CE8  = self.modified_CE(logits=self.y_logits8 , labels=self.y8 , k=8 , use_priors=self.config.priors)
                    CE6  = self.modified_CE(logits=self.y_logits6 , labels=self.y6 , k=6 , use_priors=self.config.priors)
                    CE4  = self.modified_CE(logits=self.y_logits4 , labels=self.y4 , k=4 , use_priors=self.config.priors)
                else:
                    CE10 = tf.nn.softmax_cross_entropy_with_logits(logits=self.y_logits10, labels=self.y10)
                    CE8  = tf.nn.softmax_cross_entropy_with_logits(logits=self.y_logits8 , labels=self.y8 )
                    CE6  = tf.nn.softmax_cross_entropy_with_logits(logits=self.y_logits6 , labels=self.y6 )
                    CE4  = tf.nn.softmax_cross_entropy_with_logits(logits=self.y_logits4 , labels=self.y4 )

                self.loss10 = tf.reduce_mean(c10 * CE10 + self.config.l2_beta * tf.nn.l2_loss(self.weights['w10']))
                self.loss8 =  tf.reduce_mean(c8  * CE8  + self.config.l2_beta * tf.nn.l2_loss(self.weights['w8' ]))
                self.loss6 =  tf.reduce_mean(c6  * CE6  + self.config.l2_beta * tf.nn.l2_loss(self.weights['w6' ]))
                self.loss4 =  tf.reduce_mean(c4  * CE4  + self.config.l2_beta * tf.nn.l2_loss(self.weights['w4' ]))

                self.loss = 0
                rmse_loss = tf.sqrt(tf.reduce_mean(tf.pow(self.y_pred_reg - self.y_cont, 2), name='RMSE')) * self.config.rmse_weights
                ccc_err_loss = self.ccc_err(self.y_pred_reg ,self.y_cont) * self.config.cccerr_weights
                yout_rmse_loss = tf.sqrt(tf.reduce_mean(tf.pow(self.y_out - self.y_cont, 2), name='RMSE')) * self.config.yout_weights
                y_out_cccerr_loss = self.ccc_err(self.y_out ,self.y_cont) * self.config.cccerr_weights
                if 'loss10' in self.config['losses']:
                    print('loss10 added', self.config.alpha1)
                    self.loss += self.loss10 * self.config.alpha1
                if 'loss8' in self.config['losses']:
                    print('loss8 added', self.config.alpha1)
                    self.loss += self.loss8 * self.config.alpha1
                if 'loss6' in self.config['losses']:
                    print('loss6 added', self.config.alpha1)
                    self.loss += self.loss6 * self.config.alpha1
                if 'loss4' in self.config['losses']:
                    print('loss4 added', self.config.alpha1)
                    self.loss += self.loss4 * self.config.alpha1
                if 'rmse' in self.config['losses']:
                    print('rmse loss added', self.config.alpha2)
                    self.loss += rmse_loss * self.config.alpha2
                if 'ccc_err' in self.config['losses']:
                    print('ccc_err loss added', self.config.alpha2)
                    self.loss += ccc_err_loss * self.config.alpha2
                if 'y_out_rmse' in self.config['losses']:
                    print('y_out loss added', (2-0.5*(self.config.alpha1+self.config.alpha2)))
                    self.loss += yout_rmse_loss * (2-0.5*(self.config.alpha1+self.config.alpha2))
                if 'y_out_cccerr' in self.config['losses']:
                    print('y_out ccc_err loss added', (2-0.5*(self.config.alpha1+self.config.alpha2)))
                    self.loss += y_out_cccerr_loss * (2-0.5*(self.config.alpha1+self.config.alpha2))

                tf.summary.scalar("loss", self.loss)
                
                self.merged_summary_op = tf.summary.merge_all()

            with tf.name_scope('opt'):
                self.opt = tf.train.AdamOptimizer(self.learning_rate)
                self.train_step = self.opt.minimize(self.loss,global_step=self.global_step_tensor)
                # self.train_step_rmse = self.opt.minimize(self.loss_reg,global_step=self.global_step_tensor)

                # self.grads_loss10 = list(zip(tf.gradients(self.loss10, tf.trainable_variables()), tf.trainable_variables()))
                self.grads_loss10 = self.get_valid_grads(self.opt.compute_gradients(self.loss10))
                self.grads_rmse_loss = self.get_valid_grads(self.opt.compute_gradients(rmse_loss))
                self.grads_ccc_err_loss = self.get_valid_grads(self.opt.compute_gradients(ccc_err_loss))
                self.grads_yout_rmse_loss = self.get_valid_grads(self.opt.compute_gradients(yout_rmse_loss))
                self.grads_y_out_cccerr_loss = self.get_valid_grads(self.opt.compute_gradients(y_out_cccerr_loss))

    def _initialize_weights(self):
        weights = {

            'w10': tf.get_variable("W10", shape=[self.get_final_layer_len(), 10],initializer=tf.contrib.layers.xavier_initializer()),

            'w8': tf.get_variable("W8", shape=[self.get_final_layer_len(), 8],initializer=tf.contrib.layers.xavier_initializer()),

            'w6': tf.get_variable("W6", shape=[self.get_final_layer_len(), 6],initializer=tf.contrib.layers.xavier_initializer()),

            'w4': tf.get_variable("W4", shape=[self.get_final_layer_len(), 4],initializer=tf.contrib.layers.xavier_initializer()),

            'wr': tf.get_variable("Wr", shape=[self.get_final_layer_len(), 1],initializer=tf.contrib.layers.xavier_initializer()),
            'br': tf.get_variable("br", shape=[1],initializer=tf.zeros_initializer()),

            'b': tf.get_variable("b", shape=[1],initializer=tf.zeros_initializer())
            
        }
        if self.config.clf_bias == 0:
            weights['b10'] =  tf.get_variable("b10", shape=[10],initializer=tf.contrib.layers.xavier_initializer())
            weights['b8' ] = tf.get_variable("b8", shape=[8],initializer=tf.contrib.layers.xavier_initializer())
            weights['b6' ] = tf.get_variable("b6", shape=[6],initializer=tf.contrib.layers.xavier_initializer())
            weights['b4' ] = tf.get_variable("b4", shape=[4],initializer=tf.contrib.layers.xavier_initializer())
        else:
            weights['b10'] = tf.get_variable("b10", shape=[10],initializer=tf.zeros_initializer())
            weights['b8' ] = tf.get_variable("b8" , shape=[8 ],initializer=tf.zeros_initializer())
            weights['b6' ] = tf.get_variable("b6" , shape=[6 ],initializer=tf.zeros_initializer())
            weights['b4' ] = tf.get_variable("b4" , shape=[4 ],initializer=tf.zeros_initializer())

        return weights

    def ccc(self, y_pred, y_true):
        cov_xy = tf.reduce_mean(y_pred*y_true) - (tf.reduce_mean(y_pred) * tf.reduce_mean(y_true))
        mean_x, var_x = tf.nn.moments(y_pred,0)
        mean_y, var_y = tf.nn.moments(y_true,0)
        return (2*cov_xy/(var_x + var_y + np.square(mean_x - mean_y)))

    def ccc_err(self, y_pred, y_true):
        return 1 - self.ccc(y_pred, y_true)

    def get_final_layer_len(self):
        if self.config.fc_path == 2:
            return self.config.n_fc1+self.config.n_fc2
        elif self.config.fc_path == 1:
            return self.config.n_fc
        return 2*self.config.n_hidden

    def norm_cost(self, y, y_pred, k):
        weighted_sum_y = self.get_cont_val(y,k,use_priors=False)
        weighted_sum_yp = self.get_cont_val(y_pred,k ,use_priors=False)
        return 1+tf.sqrt((tf.square(weighted_sum_y-weighted_sum_yp)))

    def get_cont_val(self, y, k, method='weighted_sum', use_priors=True):
        centers = self.data.kmeans[self.config.dim][k].cluster_centers_.flatten()
        
        if use_priors:
            y /= self.data.priors[self.config.dim][k] # divide by the k_cluster priors
            y /= tf.reshape(tf.reduce_sum(y,axis=1),[7500,1]) # renormalize

        if method == 'weighted_sum':
            return tf.reduce_sum(y*tf.constant(centers,dtype=tf.float32),axis=1)

    def norm(slef, x):
        return x / tf.reshape(tf.reduce_sum(x,1),shape=(7500,1))    

    def modified_CE(self, logits=None, labels=None, k=None, use_priors=False):
        scaled_logits = logits - tf.reshape(tf.reduce_max(logits,1),shape=(7500,1))
        normalized_logits = scaled_logits - tf.reshape(tf.reduce_logsumexp(scaled_logits,1),shape=(7500,1))

        if use_priors:
            normalized_logits -= tf.log(np.array(self.data.priors[self.config.dim][k],dtype=np.float32))
            normalized_logits -= tf.reshape(tf.reduce_logsumexp(normalized_logits,1),shape=(7500,1))

        return tf.reduce_mean(-tf.reduce_sum(labels*normalized_logits,1))
