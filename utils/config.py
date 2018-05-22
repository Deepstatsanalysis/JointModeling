import json
from bunch import Bunch
import os


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace) or config(dictionary)
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        config_dict = json.load(config_file)

    # convert the dictionary to a namespace using bunch lib
    config = Bunch(config_dict)
    config = default_values(config)

    return config, config_dict


def process_config(jsonfile):
    config, _ = get_config_from_json(jsonfile)
    config.summary_dir = os.path.join("../experiments", config.exp_name, "summary")
    config.checkpoint_dir = os.path.join("../experiments", config.exp_name, "checkpoint")
    return config

def default_values(config):

    config['target_cluster']       = -1                 if not 'target_cluster'      in config.keys() else config['target_cluster'] 
    config['rater_id']             = -1                 if not 'rater_id'            in config.keys() else config['rater_id'] 
    config['gt_priors']            = False              if not 'gt_priors'           in config.keys() else config['gt_priors']
    config['priors']               = False              if not 'priors'              in config.keys() else config['priors']
    config['reg']                  = False              if not 'reg'                 in config.keys() else config['reg']
    config['modified_CE']          = False              if not 'modified_CE'         in config.keys() else config['modified_CE']
    config['ccc_err']              = False              if not 'ccc_err'             in config.keys() else config['ccc_err']
    config['rmse_weights']         = 1                  if not 'rmse_weights'        in config.keys() else config['rmse_weights']
    config['cccerr_weights']       = 1                  if not 'cccerr_weights'      in config.keys() else config['cccerr_weights']
    config['yout_weights']         = 1                  if not 'yout_weights'        in config.keys() else config['yout_weights']
    config['alpha1']               = 1                  if not 'alpha1'              in config.keys() else config['alpha1']
    config['alpha2']               = 1                  if not 'alpha2'              in config.keys() else config['alpha2']
    config['fcs_num']              = 0                  if not 'fcs_num'             in config.keys() else config['fcs_num']
    config['n_fc']                 = 16                 if not 'n_fc'                in config.keys() else config['n_fc']
    config['fc_act']               = 'tanh'             if not 'tanh'                in config.keys() else config['tanh']
    config['fc_path']              = 0                  if not 'fc_path'             in config.keys() else config['fc_path']
    config['clf_bias']             = 0                  if not 'clf_bias'            in config.keys() else config['clf_bias']
    config['audio_video_feat']     = 0                  if not 'audio_video_feat'    in config.keys() else config['audio_video_feat']
    config['clf_bias']             = 1                  if not 'clf_bias'            in config.keys() else config['clf_bias']
    config['gt']                   = 'onehot'           if not 'gt'                  in config.keys() else config['gt']
    config['ccc_diff']             = -0.01              if not 'ccc_diff'            in config.keys() else config['ccc_diff']
    config['reset_lr']             = True               if not 'reset_lr'            in config.keys() else config['reset_lr']
    config['stage2']               = 0                  if not 'stage2'              in config.keys() else config['stage2']
    config['max_to_keep']          = 1000               if not 'max_to_keep'         in config.keys() else config['max_to_keep']
    config['subset']               = 'joint_modling'    if not 'subset'              in config.keys() else config['subset']
    config['log_dir']              = 'logs'             if not 'log_dir'             in config.keys() else config['log_dir']
    config['max_length']           = 7500               if not 'max_length'          in config.keys() else config['max_length']
    config['sequence_length']      = 7500               if not 'sequence_length'     in config.keys() else config['sequence_length']
    config['learning_rate']        = 0.02               if not 'learning_rate'       in config.keys() else config['learning_rate']
    config['num_epochs']           = 20                 if not 'num_epochs'          in config.keys() else config['num_epochs']

    return config
