# Joint Modeling
Source code for Joint Modeling, created by Ehab AlBadawy at UAlbany

### Contents
1. [Requirements](#requirements)
2. [Usage](#usage)
3. [Instructions](#instructions)

### Requirements

You need `TensorFlow`. Please refer to [TensorFLow instalation instructions](https://www.tensorflow.org/install/)

### Usage

#### Folder structure
--------------

```
├──  base
│   ├── base_model.py   - this file contains the abstract class of the model.
│   └── base_train.py   - this file contains the abstract class of the trainer.
│   └── base_test.py    - this file contains the abstract class of the tester.
│
│
├── model               - this folder contains the BLSTM model implementation.
│   └── BLSTM.py
│
│
├── trainer             - this folder contains trainers classes for stage1 and stage2.
│   └── trainer.py
│   └── trainer_stage2.py
│   
├──  mains              - here's the main to run your experiments based on the config file.
│    └── main.py        - main file that is responsible for the whole pipeline.
│  
├──  data _loader  
│    └── data_generator.py  - here's the data_generator that is responsible for all data handling.
│ 
└── utils
     ├── logger.py
     └── metrics.py
     └── config.py
     └── utils.py

```

#### config file

the configuration file is used to tell Tensorflow what hyper-paramters you want to use for the BLSTM model.

##### Parameters


**exp_name**: This paramter defines the name of your experment. the value is a string.

**init_exp**: This paramter indicates when the network should stop training. The value is a string.

**num_epochs**: [default = 20] This paramter idicates the number of epochs for the network to be trained. The value is an integer.

**learning_rate**: [default = 0.02] This parameter indicates the base (beginning) learning rate of the network. The value is a real number (floating ponit).

**n_input**: This paramter indicates the number of inputs for the BLSTM model. the value is an integer.

**n_hidden**: This paramter indicates the number hidden units for the BLSTM model. the value is an integer.

**n_layers**: This paramter indicates number of layers for the BLSTM model. the value is an integer.

**max_length**: [default = 7500] This paramter indicates the number of frames the BLSTM model should use. the value is an integer.

**log_dir**: [default = 'logs'] This paramter holds the log director name where the tensorboard logs should be saved. The value is a string.

**subset**: [default = 'joint_modeling'] This paramter holds the sub director for he tensorboard logs. The value is a string.

**max_to_keep**: [default = 1000] This paramters indicates the maximum number of models to be save. The value is integer.

**dim**: This paramters indicates what dimension to be used while traing/testing. The value is string. Possible values (`arousal`, `valence`)

**sequence_length**: This paramters indicates input squence length for the BLSTM model. The value is an integer.

**dataset_root**: This paramters holds the path to the dataset root. This directory should include the input features, one-hot encoded GT, and the k-means models as defined in `data_generator.py` init function. The value is a string.

**recola_root**: This paramters holds the path to RECOLA root to load the continuous GT as defined in `get_get_cont` function in `data_generator.py` class. The value is a string.

**stage2**: [default = 0] this paramters indicates if the current traing is for the 2nd stage. The value is boolean.

**cost**: This paramters indicates what cost function is used. The value is string. Possible values (`argmax`, `norm`)

**ccc_diff** [default = -0.01] This paramter indicates the difference between current CCC value and the best CCC found in the 2nd stage, if the difference is lower that `ccc_diff`, the learning rate will be havled. The value is real number (float).

**reset_lr**: [default = True] This paramter is a flag for eaither resting the learning rate to the base learning rate `learning_rate` after a good CCC found in the 2nd stage or keep the current learning rate. The vale is boolean.

**modified_CE**: [defaults = False] This paramter is a flag to either use the modified version of the cross entropy loss (set True if you want to apply priors on the network predictions). The value is boolean.

**audio_video_feat**: [defaults = False] This paramter is a flag for either to use the video features or not. The value is boolean.

**alpha1**:[default = 1] This paramter holds the value of alpha1 in end-to-end model loss equation. The value is real number (float). Take values between 0 and 1.

**alpha2**:[default = 1] This paramter holds the value of alpha2 in end-to-end model loss equation. The value is real number (float). Take values between 0 and 1.

**fc_act**: [default = 'tanh'] This paramter holds the type of the FC layer activation function. The value is string. Possible values ('tanh', 'sigmoid')

**fcs_num**: [default = 0] This paramter holds the number of the FC layers after BLSTM model. The value is integer. Note: set to 0 to ignore the FC layer.

**n_fc**: [default = 16] This paramter holds number of FC units. The value is integer.

**target_ccc**: This paramter indicates the current CCC to optimize for the 2nd stage. The value is a string. Possible values ('argmax', 'argmax_priors', 'weights_sum', 'weighted_sum_priots', 'reg')

**losses**: This paramter holds the losses to be used for training. The value is array of strings. Possible values :
- 'loss4': loss for k-means clustring where k = 4.
- 'loss6': loss for k-means clustring where k = 6.
- 'loss8': loss for k-means clustring where k = 8.
- 'loss10': loss for k-means clustring where k = 10.
- 'rmse': RMAE loss for regreesion node.
- 'ccc_err': CCC loss for the regression node.
- 'y_out_rmse': RMSE loss for the final node.
- 'y_out_cccerr': CCC loss for the final node.

Note: the model will only consider the losses that are exsit in the array (e.g. set `losses: ['loss4', 'loss6', 'loss8', 'loss10']` if you want to train the model for classifcation task only using the 4 k-means models). 




### Instructions

To train your model, you need to define the model hyper-paramters in config file (e.g. `configs/example.json`) and run the following cmd from `mains` director

```
python main.py --config=../configs/example.json # replace example.json with your config file
```
