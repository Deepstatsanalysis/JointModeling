Folder structure
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
