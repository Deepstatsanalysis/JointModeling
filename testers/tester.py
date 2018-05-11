from base.base_test import BaseTester
from tqdm import tqdm
import numpy as np

from utils import metrics

class Tester(BaseTester):
    def __init__(self, sess, model, data, config):
        super(Tester, self).__init__(sess, model, data, config)

    def test(self, subset):
        loop = tqdm(range(9))
        y_pred_all = {}
        y_pred_all['argmax'] = []
        y_pred_all['weighted_sum'] = []
        y_pred_all['argmax_priors'] = []
        y_pred_all['weighted_sum_priors'] = []
        y_pred_all['reg'] = []
        y_pred_all['y_out'] = []
        losses = []

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

        if subset == 'test':
            ccc = loss =  y_gt = None
        else:
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

        return ccc, loss, y_pred_all, y_gt

    def step(self, dim, subset, idx):
        batch_x, y10, y8, y6, y4 = next(self.data.get_batch(self.config.dim, subset, idx))
        if subset == 'test':
            y_cont = self.data.get_gt_cont(dim, 'dev', idx)
        else:
            y_cont = self.data.get_gt_cont(dim, subset, idx)


        feed_dict = {self.model.X: batch_x, self.model.y_cont: y_cont, self.model.y10: y10, self.model.y8: y8, self.model.y6: y6, self.model.y4: y4,
                                    self.model.is_training: False}

        loss, y_reg, y_out, yp10, yp8, yp6, yp4 = self.sess.run([self.model.loss, self.model.y_pred_reg, self.model.y_out, self.model.y_pred10, 
                self.model.y_pred8, self.model.y_pred6, self.model.y_pred4], feed_dict=feed_dict)

        pred = {}
        pred['argmax'] = self.pred_argmax(yp10, yp8, yp6, yp4, dim, priors=False)
        pred['weighted_sum'] = self.pred_weighted_sum(yp10, yp8, yp6, yp4, dim, priors=False)

        pred['argmax_priors'] = self.pred_argmax(yp10, yp8, yp6, yp4, dim, priors=True)
        pred['weighted_sum_priors'] = self.pred_weighted_sum(yp10, yp8, yp6, yp4, dim, priors=True)
        pred['reg'] = y_reg
        pred['y_out'] = y_out
        
        return loss, pred