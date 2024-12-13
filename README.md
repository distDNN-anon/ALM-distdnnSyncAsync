# FL and DDP synchronous and asynchronous versions as derived methods of our general mathematical framework: Aggregative Learning Model

Federated Learning and Distributed Data Parallelism implementation in both synchronous and asynchronous modes. 

install required dependencies using pip utility:

`` pip install requirements.txt `` 

Training in distributed mode, e.g. : runing with mpirun tools, example distributed training with 2 processes (2 GPUs)
    
## Case Image classification (Tiny ImageNet dataset) - the code in root directory

`` mpirun -np 2 python test_ddp_async.py --model-name 'resnet18'``

`` mpirun -np 2 python test_ddp_async.py --model-name 'efficientnet_b0' ``

## Case Natural Language Understanding: The code is located in the NLU directory. 
Please navigate to this directory before running the script.

`` mpirun -np 2 python test_nlp_ddp_async.py --model-name 'bert-base-uncased'``

### Training in sequential mode, only use one process, you need only to replace the 2 by 1. 
--------------------------------------------------------------------------------------------

You can use the code for image classification with other datasets (e.g., ImageNet-1k) or models. A few adjustments are required: download the dataset and set its path in the utils.py file. For any model (pretrained or not), simply reference the model name placed in the pretrained_models folder.
For NLU tasks, a few settings are also required in the corresponding utils.py file located in the NLU folder. You must specify the dataset and model to be used.






