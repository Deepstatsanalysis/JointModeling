from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np
import math
import os
from utils import metrics

class TrainerStage2(BaseTrain):
    def __init__(self, sess, model, data, config,logger):
        super(TrainerStage2, self).__init__(sess, model, data, config,logger)

    def train(self):
        config_path = '../configs'
        cur_epoch = self.model.cur_epoch_tensor.eval(self.sess)
        ccc_dev, loss = self.epoch('dev')
        self.log(ccc_dev, self.config.target_ccc, loss, 'dev', cur_epoch*9)

        self.best_ccc = ccc_dev[self.config.target_ccc]
        
        while cur_epoch < self.config.num_epochs and self.learning_rate > 1e-05:
            print('epoch:', self.model.cur_epoch_tensor.eval(self.sess),', iter:', cur_epoch*9, '(%d)'%self.model.global_step_tensor.eval(self.sess))
            ccc_train, loss_train = self.epoch('train', learning_rate=self.learning_rate)
            ccc_dev, loss_dev = self.epoch('dev')

            print(ccc_dev[self.config.target_ccc], self.best_ccc)

            if math.isnan(self.best_ccc):
                os.system('echo "python main.py --config=%s/%s.json && python main.py --config=%s/%s.json" >> failed_exps'%(config_path, self.config.init_exp, config_path, self.config.exp_name))
                break

            if ccc_dev[self.config.target_ccc] - self.best_ccc > self.config.ccc_diff:
                self.sess.run(self.model.increment_cur_epoch_tensor)
                cur_epoch += 1
                
                
                self.best_iter = self.model.global_step_tensor.eval(self.sess)
                
                if ccc_dev[self.config.target_ccc] > self.best_ccc: 
                    self.best_ccc = ccc_dev[self.config.target_ccc]
                    
                self.model.save(self.sess, '../weights/%s/model'%(self.config.exp_name))


                self.log(ccc_dev, self.config.target_ccc, loss_dev, 'dev', cur_epoch*9)
                self.log(ccc_train, self.config.target_ccc, loss_train, 'train', cur_epoch*9, self.learning_rate)

                if self.config.reset_lr:
                    self.learning_rate = np.array(self.config.learning_rate)
            else:
                self.model.load(self.sess, '../weights/%s/model-%d'%(self.config.exp_name,self.best_iter))
                self.learning_rate = self.learning_rate / 2
                print('halving the lr to be %f' % self.learning_rate)
                

    def epoch(self, subset, learning_rate=0.002):
        loop = tqdm(range(9))
        losses=[]
        y_pred_all = {}
        y_pred_all['argmax'] = []
        y_pred_all['weighted_sum'] = []
        y_pred_all['argmax_priors'] = []
        y_pred_all['weighted_sum_priors'] = []
        y_pred_all['reg'] = []
        y_pred_all['y_out'] = []
        for idx, it in enumerate(loop):
            loss, y_pred = self.step(self.config.dim, subset, idx+1, learning_rate)
            losses.append(loss)
            y_pred_all['argmax'].append(y_pred['argmax'])
            y_pred_all['weighted_sum'].append(y_pred['weighted_sum'])
            y_pred_all['argmax_priors'].append(y_pred['argmax_priors'])
            y_pred_all['weighted_sum_priors'].append(y_pred['weighted_sum_priors'])
            y_pred_all['reg'].append(y_pred['reg'])
            y_pred_all['y_out'].append(y_pred['y_out'])
            
        y_pred_all['argmax'] = np.array(y_pred_all['argmax']).flatten()
        y_pred_all['weighted_sum'] = np.array(y_pred_all['weighted_sum']).flatten()
        y_pred_all['argmax_priors'] = np.array(y_pred_all['argmax_priors']).flatten()
        y_pred_all['weighted_sum_priors'] = np.array(y_pred_all['weighted_sum_priors']).flatten()
        y_pred_all['reg'] = np.array(y_pred_all['reg']).flatten()
        y_pred_all['y_out'] = np.array(y_pred_all['y_out']).flatten()

        if self.config.rater_id > 0:
            y_gt = self.data.load_combined_gt_rater_specific(self.config.dim, [subset], self.config.rater_id).flatten()
        else:
            y_gt = self.data.load_combined_gt(self.config.dim, [subset]).flatten()


        loss=np.mean(losses)
        ccc = {}
        ccc['argmax'] = metrics.ccc(y_pred_all['argmax'], y_gt)
        ccc['weighted_sum'] = metrics.ccc(y_pred_all['weighted_sum'], y_gt)
        ccc['argmax_priors'] = metrics.ccc(y_pred_all['argmax_priors'], y_gt)
        ccc['weighted_sum_priors'] = metrics.ccc(y_pred_all['weighted_sum_priors'], y_gt)
        ccc['reg'] = metrics.ccc(y_pred_all['reg'], y_gt)
        ccc['y_out'] = metrics.ccc(y_pred_all['y_out'], y_gt)
        y_pred_avg = np.sum([4*y_pred_all['weighted_sum'],y_pred_all['reg']],axis=0)/5
        ccc['avg'] = metrics.ccc(y_pred_avg, y_gt)

        return ccc, loss


    def step(self, dim, subset, idx, learning_rate):
        batch_x, y10, y8, y6, y4 = next(self.data.get_batch(self.config.dim, subset, idx))
        y_cont = self.data.get_gt_cont(dim, subset, idx)

        feed_dict = {self.model.X: batch_x, self.model.y_cont: y_cont, self.model.y10: y10, self.model.y8: y8, self.model.y6: y6, self.model.y4: y4, self.model.learning_rate: self.config.learning_rate}
        
        if subset == 'train':        
            feed_dict[self.model.learning_rate] = learning_rate
            feed_dict[self.model.is_training] = True
            _, loss, y_reg, y_out, yp10, yp8, yp6, yp4,img_summary = self.sess.run([self.model.train_step, self.model.loss, self.model.y_pred_reg, self.model.y_out,
                self.model.y_pred10, self.model.y_pred8, self.model.y_pred6, self.model.y_pred4, self.model.img_summary_op], feed_dict=feed_dict)
        else:
            feed_dict[self.model.is_training] = False

            loss, y_reg, y_out, yp10, yp8, yp6, yp4,img_summary = self.sess.run([self.model.loss, self.model.y_pred_reg, self.model.y_out, self.model.y_pred10, 
                self.model.y_pred8, self.model.y_pred6, self.model.y_pred4, self.model.img_summary_op], feed_dict=feed_dict)

        pred = {}
        pred['argmax'] = self.pred_argmax(yp10, yp8, yp6, yp4, dim, priors=False)
        pred['weighted_sum'] = self.pred_weighted_sum(yp10, yp8, yp6, yp4, dim, priors=False)

        pred['argmax_priors'] = self.pred_argmax(yp10, yp8, yp6, yp4, dim, priors=True)
        pred['weighted_sum_priors'] = self.pred_weighted_sum(yp10, yp8, yp6, yp4, dim, priors=True)
        pred['reg'] = y_reg
        pred['y_out'] = y_out
        
        return loss, pred

    def log(self, ccc, target_ccc, loss, subset, cur_it, lr=-1):
        summaries_dict = {}
        summaries_dict['loss'] = loss
        summaries_dict['ccc_argmax'] = ccc['argmax']
        summaries_dict['ccc_weighted_sum'] = ccc['weighted_sum']
        summaries_dict['ccc_argmax_priors'] = ccc['argmax_priors']
        summaries_dict['ccc_weighted_sum_priors'] = ccc['weighted_sum_priors']
        summaries_dict['ccc_reg'] = ccc['reg']
        summaries_dict['ccc_avg'] = ccc['avg']
        summaries_dict['ccc_y_out'] = ccc['y_out']

        print(ccc[self.config.target_ccc], lr)
        if lr > 0:
            summaries_dict['lr'] = lr
        self.logger.summarize(cur_it, summaries_dict=summaries_dict,summerizer=subset)