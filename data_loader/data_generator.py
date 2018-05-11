import numpy as np
import os

class DataGenerator:
    def __init__(self, config):
        self.config = config

        self.input = {}
        if self.config.audio_video_feat:
            self.input['train'] = np.load('%s/audio_vido_feat_train_norm.npy'%config.dataset_root).item()
            self.input['dev'] = np.load('%s/audio_vido_feat_dev_norm.npy'%config.dataset_root).item()
            self.input['test'] = np.load('%s/audio_vido_feat_test_norm.npy'%config.dataset_root).item()
        else:
            self.input['train'] = np.load('%s/train_reshape_norm.npy'%config.dataset_root).item()
            self.input['dev'] = np.load('%s/dev_reshape_norm.npy'%config.dataset_root).item()
            self.input['test'] = np.load('%s/test_reshape_norm.npy'%config.dataset_root).item()
        self.priors = np.load('%s/priors.npy'%config.dataset_root).item() # priors[dim][k]

        self.y = np.load('%s/gt_%s.npy'%(config.dataset_root,self.config.gt)).item() # y[dim][subset][idx][k]
        if self.config.gt_priors:
            self.y = self.apply_priors_on_gt(self.y)

        self.kmeans = np.load('%s/kmeans_ordered.npy'%config.dataset_root).item() # kmeans[subset][k]


    def get_batch(self, dim, subset, idx):
        x = self.input[subset]['%s_%d'%(subset, idx)].T.reshape((1,self.config.sequence_length,self.config.n_input))
        
        if subset == 'test': # load dev GT instead when getting test set prediction
            subset = 'dev'
        y10 = self.y[dim][subset][idx][10].reshape((self.config.sequence_length,10))
        y8 = self.y[dim][subset][idx][8].reshape((self.config.sequence_length,8))
        y6 = self.y[dim][subset][idx][6].reshape((self.config.sequence_length,6))
        y4 = self.y[dim][subset][idx][4].reshape((self.config.sequence_length,4))

        yield x, y10, y8, y6, y4


    def get_gt_cont(self, dim, subset, idx):
        """ Loads GT for a specific session
        Args:
            dim: GT dimension, take values of 'arousal', or 'valence'
            subset: dataset of the returned GT, take values of 'train', 'dev'
            idx: session index, take values [1-9]

        Returns:
            numpy array of size (7500, 1) for the requested GT based on the args

        """
        gt = os.popen('cat %s/%s/%s_%d.arff'%(self.config.recola_root, dim, subset, idx)).read()
        gt = gt.split('\n')[9:] # remove the heading attr
        gt = gt[:7500] # get the first 7500 gtures
        gt = [float(gt[i].split(',')[-1]) for i in range(len(gt))] # get the 3rd col for each raw as float
        
        gt = np.array(gt)
        return gt.reshape(gt.shape[0],1)

    def load_combined_gt(self, dim, subsets):
        """ Loads GT for a combined subsets
        Args:
            dim: GT dimension, take values of 'arousal', or 'valence'
            subsets: list of the returend GT subsets, i.e. ['train'] or ['train', 'dev']

        Returns:
            numpy array of size (67500, 1) for the requested GT based on the args
        """
        
        gt = []
        
        for subset in subsets:
            for idx in range(1,10):
                gt.append(self.get_gt_cont(dim, subset, idx))
               
        gt = np.array(gt).flatten()
        return gt.reshape(gt.shape[0],1)

    def norm(self, pred):
        return pred / np.sum(pred,axis=1,keepdims=1)

    def apply_priors_on_gt(self,gt):
        # gt[dim][subset][idx][k]
        dim = self.config.dim
        for subset in ['train', 'dev']:
            for idx in range(1,10):
                for k in [4,6,8,10]:
                    gt[dim][subset][idx][k] = self.norm(gt[dim][subset][idx][k] / self.priors[dim][k])
        return gt
