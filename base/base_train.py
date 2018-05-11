import tensorflow as tf


class BaseTrain:
    def __init__(self, sess, model, data, config,logger):

        self.config = config
        self.model = model
        self.logger = logger
        self.sess = sess
        self.data = data
        self.best_ccc = 0
        self.best_iter = self.config.init_epoch*9
        self.learning_rate = np.array(self.config.learning_rate)

        self.init_var()
        
        my_mkdir('../weights')
        my_mkdir('../weights/%s'%(self.config.exp_name))

    def init_var(self):
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        self.sess.run(init)

    def train(self):
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.num_epochs + 1, 1):
            self.epoch('train')
            self.epoch('dev')
            self.sess.run(self.model.increment_cur_epoch_tensor)
            self.model.save(self.sess, '../weights/%s/model'%(self.config.exp_name))

    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError
