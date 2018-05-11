import tensorflow as tf


class BaseModel:
    def __init__(self, config, data):
        self.config = config
        self.data = data
        # init the global step
        self.init_global_step_val = self.config.init_epoch*9
        self.init_global_step(self.init_global_step_val)
        # init the epoch counter
        self.init_cur_epoch()

    # save function thet save the checkpoint in the path defined in 'outdir' var
    def save(self, sess, outdir):
        print("Saving model...")
        self.saver.save(sess, outdir, self.global_step_tensor)
        print("Model saved")

    # load checkpoint from the experiment path defined in 'model' var
    def load(self, sess, model):
        print("Loading model checkpoint {} ...\n".format(model))
        self.saver.restore(sess, model)
        print("Model loaded")

    # just initialize a tensorflow variable to use it as epoch counter
    def init_cur_epoch(self):
        with tf.variable_scope('cur_epoch'):
            self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
            self.increment_cur_epoch_tensor = tf.assign(self.cur_epoch_tensor, self.cur_epoch_tensor + 1)

    # just inialize a tensorflow variable to use it as global step counter
    def init_global_step(self, init_value):
        # DON'T forget to add the global step tensor to the tensorflow trainer
        with tf.variable_scope('global_step'):
            self.global_step_tensor = tf.Variable(init_value, trainable=False, name='global_step')

    def init_saver(self):
        # just copy the following line in your child class
        # self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)
        raise NotImplementedError

    def build_model(self):
        raise NotImplementedError
