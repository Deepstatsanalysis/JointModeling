import numpy as np


class DataGenerator:
    def __init__(self, config):
        self.config = config
        # load data here
        self.input = np.ones((500, 784))
        self.y = np.ones((500, 10))

    def get_batch(self, dim, subset, idx):
        x = self.input[subset]['%s_%d'%(subset, idx)].T.reshape((1,self.config.sequence_length,self.config.n_input))
        
        if subset == 'test': # load dev GT instead when getting test set prediction
            subset = 'dev'
        y10 = self.y[dim][subset][idx][10].reshape((self.config.sequence_length,10))
        y8 = self.y[dim][subset][idx][8].reshape((self.config.sequence_length,8))
        y6 = self.y[dim][subset][idx][6].reshape((self.config.sequence_length,6))
        y4 = self.y[dim][subset][idx][4].reshape((self.config.sequence_length,4))


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
        
    def next_batch(self, batch_size):
        idx = np.random.choice(500, batch_size)
        yield self.input[idx], self.y[idx]
