from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np

from utils import metrics

class Trainer(BaseTrain):
    def __init__(self, sess, model, data, config,logger):
        super(Trainer, self).__init__(sess, model, data, config,logger)

    def epoch(self, subset):
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
            loss, y_pred = self.step(self.config.dim, subset, idx+1)
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
            y_gt = self.data.load_combined_gt(self.config.dim, [subset]).flatten()#[0:7500]

        loss=np.mean(losses)
        ccc_argmax = metrics.ccc(y_pred_all['argmax'], y_gt)
        ccc_weighted_sum = metrics.ccc(y_pred_all['weighted_sum'], y_gt)
        ccc_argmax_priors= metrics.ccc(y_pred_all['argmax_priors'], y_gt)
        ccc_weighted_sum_priors = metrics.ccc(y_pred_all['weighted_sum_priors'], y_gt)
        ccc_reg = metrics.ccc(y_pred_all['reg'], y_gt)
        ccc_y_out = metrics.ccc(y_pred_all['y_out'], y_gt)

        y_pred_avg = np.sum([4*y_pred_all['weighted_sum'],y_pred_all['reg']],axis=0)/5
        ccc_avg = metrics.ccc(y_pred_avg, y_gt)

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {}
        summaries_dict['loss'] = loss
        summaries_dict['ccc_argmax'] = ccc_argmax
        summaries_dict['ccc_weighted_sum'] = ccc_weighted_sum
        summaries_dict['ccc_argmax_priors'] = ccc_argmax_priors
        summaries_dict['ccc_weighted_sum_priors'] = ccc_weighted_sum_priors
        summaries_dict['ccc_reg'] = ccc_reg
        summaries_dict['ccc_y_out'] = ccc_y_out
        summaries_dict['ccc_avg'] = ccc_avg

        print('loss: %0.4f, ccc_argmax: %0.4f, ccc_weighted_sum: %0.4f, ccc_argmax_priors: %0.4f, ccc_weighted_sum_priors: %0.4f, ccc_reg: %0.4f, ccc_y_out: %0.4f, ccc_avg: %0.4f'%(loss,ccc_argmax,ccc_weighted_sum,ccc_argmax_priors,ccc_weighted_sum_priors,ccc_reg,ccc_y_out,ccc_avg))
        

        self.logger.summarize(cur_it, summaries_dict=summaries_dict,summerizer=subset)

    def step(self, dim, subset, idx):
        batch_x, y10, y8, y6, y4 = next(self.data.get_batch(self.config.dim, subset, idx))
        y_cont = self.data.get_gt_cont(dim, subset, idx)

        feed_dict = {self.model.X: batch_x, self.model.y_cont: y_cont, self.model.y10: y10, self.model.y8: y8, self.model.y6: y6, self.model.y4: y4, self.model.learning_rate: self.config.learning_rate}


        if subset == 'train':        
        
            feed_dict[self.model.is_training] = True
            _, loss, y_reg, y_out , yp10, yp8, yp6, yp4, img_summary = self.sess.run([self.model.train_step, self.model.loss, self.model.y_pred_reg, self.model.y_out,
                self.model.y_pred10, self.model.y_pred8, self.model.y_pred6, self.model.y_pred4, self.model.img_summary_op], feed_dict=feed_dict)
        else:
            feed_dict[self.model.is_training] = False
            loss, y_reg, y_out, yp10, yp8, yp6, yp4, img_summary = self.sess.run([self.model.loss, self.model.y_pred_reg, self.model.y_out, self.model.y_pred10,
                self.model.y_pred8, self.model.y_pred6, self.model.y_pred4, self.model.img_summary_op], feed_dict=feed_dict)

        pred = {}
        pred['argmax'] = self.pred_argmax(yp10, yp8, yp6, yp4, dim, priors=False)
        pred['weighted_sum'] = self.pred_weighted_sum(yp10, yp8, yp6, yp4, dim, priors=False)

        pred['argmax_priors'] = self.pred_argmax(yp10, yp8, yp6, yp4, dim, priors=True)
        pred['weighted_sum_priors'] = self.pred_weighted_sum(yp10, yp8, yp6, yp4, dim, priors=True)
        pred['reg'] = y_reg
        pred['y_out'] = y_out
        return loss, pred