import tensorflow as tf

import os
import sys
sys.path.append('../')

from data_loader.data_generator import DataGenerator
from models.BLSTM import BLSTM
from testers.tester import Tester
from trainers.trainer import Trainer
from trainers.trainer_stage2 import TrainerStage2
from trainers.trainer_reg import TrainerReg
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args, get_best_epoch


def main():
    # capture the config path from the run arguments
    # then process the json configration file
    try:
        args = get_args()
        config = process_config(args.config)

    except:
        print("missing or invalid arguments")
        exit(0)
    
    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
    sess = tf.Session()
    # create data generator instance
    data = DataGenerator(config)
    # create model instance
    model = BLSTM(config, data)
    # create tensorboard logger
    logger = Logger(sess, config)

    if config.stage2:
        config.init_epoch = get_best_epoch(config)
        trainer = TrainerStage2(sess, model, data, config, logger)
        # load init weights and save them in the new exp dir
        trainer.model.load(sess, '../weights/%s/model-%d'%(config.init_exp, config.init_epoch*9))
        trainer.model.save(sess, '../weights/%s/model'%(config.exp_name))
    else:
        # create trainer and pass all previous components to it
        trainer = Trainer(sess, model, data, config, logger)

    # here the model is training
    trainer.train()

if __name__ == '__main__':
    main()