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

    def next_batch(self, batch_size):
        idx = np.random.choice(500, batch_size)
        yield self.input[idx], self.y[idx]
