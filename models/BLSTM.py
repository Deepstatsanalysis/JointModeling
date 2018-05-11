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
