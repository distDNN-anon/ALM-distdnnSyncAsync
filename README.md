# FL and DDP synchronous and asynchronous versions as derived methods of our general mathematical framework: Aggregative Learning Model

Federated Learning and Distributed Data Parallelism implementation in both synchronous and asynchronous modes. 

Download the dataset tiny-imagenet-200 from  and put it into the datasets directory

Run the training, e.g. : runing with mpirun tools, example with 4 processes (4 GPUs)

`` mpirun -np 4 python test_ddp_async.py --model-name 'resnet18'``

`` mpirun -np 4 python test_ddp_async.py --model-name 'efficientnet_b0' ``



