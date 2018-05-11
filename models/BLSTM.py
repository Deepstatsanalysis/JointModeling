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