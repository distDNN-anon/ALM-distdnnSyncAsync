# FL and DDP synchronous and asynchronous versions as derived methods of our general mathematical framework: Aggregative Learning Model

Federated Learning and Distributed Data Parallelism implementation in both synchronous and asynchronous modes. 

install required dependencies using pip utility:

`` pip install requirements.txt `` 

Training in distributed mode, e.g. : runing with mpirun tools, example distributed training with 2 processes (2 GPUs)

`` mpirun -np 2 python test_ddp_async.py --model-name 'resnet18'``

`` mpirun -np 2 python test_ddp_async.py --model-name 'efficientnet_b0' ``

Training in sequential mode, only use one process (1 GPU): 

`` mpirun -np 1 python sequential.py --model-name 'resnet18'``

`` mpirun -np 1 python sequential.py --model-name 'efficientnet_b0' ``




