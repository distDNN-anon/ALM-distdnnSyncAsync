import time
import torch.nn as nn
import argparse
import numpy as np
from utils import *
from mpi4py import MPI
import sys


device = torch.device('cuda')
d_cpu = torch.device('cpu')
# loss function
criterion = nn.CrossEntropyLoss()


def train(comm, args, dograd, model, optimizer, train_loader, test_loader,epoch, history):
    start_time = time.time()
    model.train()
    train_total = 0
    train_total_acc = 0
    train_total_loss = 0
    for iter, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        if iter  % args.iter_exchange == 0:
            dograd[args.rank] = True
        else:
            dograd[args.rank] = False
            
        for i in range(args.world_size):
            if i!=args.rank:        
                while(comm.iprobe(source=i, tag=100000+i)==True):
                    min_epoch[i]=comm.recv(source=i,tag=100000+i)                  
            
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)        

        loss.backward()

        # receive grad among the workers and average them, by Iterating over the parameters of the model in reverse order
        l = len(list(model.parameters()))
        for param in reversed(list(model.parameters())):
            for i in range(args.world_size):
                if i!=args.rank:
                    while(comm.iprobe(source=i, tag=i+10*args.rank + l-1)==True):
                        buff[i]=comm.recv(source=i,tag=i+10*args.rank + l-1)
                        param.grad += torch.tensor(buff[i][l-1]).to(device)
            param.grad /= args.world_size
            l-=1

        optimizer.step() 

        # send epoch to workers to notify current epoch
        if iter  % args.iter_exchange == 0:
            for i in range(args.world_size):
                if i!=args.rank:
                    comm.isend(epoch,dest=i,tag=100000+args.rank)
            min_epoch[args.rank]=epoch           
   
        
        m_epoch=np.min(min_epoch)        

        train_pred = output.max(1, keepdim=True)[1]
        train_correct = train_pred.eq(target.view_as(train_pred)).sum().item()
        train_total_acc +=  train_correct
        train_total_loss += loss.item()

        train_total += target.shape[0]

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('[Process-{}]Train Epoch {} <Loss: {:.6f}, Accuracy: {:.2f}%> total time {:3.2f}s'.format(args.rank + 1, epoch, train_total_loss / train_total , 100. * train_total_acc / train_total,elapsed_time))
    history["train_time"][epoch-1] = elapsed_time
    history["train_acc"][epoch-1] = 100. * train_total_acc / train_total
    history["train_loss"][epoch-1] = train_total_loss / train_total

    # Validation
    #if args.rank == 0:
    with torch.no_grad():
        correct = 0
        total = 0
        model.eval()
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs
        history["test_acc"][epoch - 1] = correct / total
        print('[INFOS-Process_{}] - Accuracy of the network on the {} validation images: {} %'.format(args.rank + 1, len(test_loader)*args.batch_size, 100 * correct / total)) 
        sys.stdout.flush()   

    if(m_epoch==args.epochs):
        print("[INFOS-Process_{}] - Total training time: {:.2f}" .format(args.rank + 1, sum(history["train_time"])))
        torch.save(history, './log/' + 'Async_DDP_world_' + str(args.world_size) + '_worker_' + str( args.rank + 1) + '_' + args.model_name + '_history.pt')
        sys.exit(0)


def worker(comm, args, model):
    dograd = [False for _ in range(args.world_size)]
    hookB = []
    class HookTensor():
        def __init__(self, tensor, count):
            self.hook = tensor.register_hook(self.hook_fn)
            self.count=count
            
        def hook_fn(self, tensor):
            if dograd[args.rank]:
                tensor=tensor.to(d_cpu)
                t2=tensor.clone().detach().cpu().numpy()
                
                mygrad[self.count]=t2

                for i in range(args.world_size):
                    if i!=args.rank:
                        if(req_s[i]==None):
                            time.sleep(0.01)
                            req_s[i]=comm.isend(mygrad,dest=i,tag=args.rank+10*i + self.count)
                        else:
                            v = req_s[i].Test()
                            if(v==True):
                                time.sleep(0.01)
                                req_s[i]=comm.isend(mygrad,dest=i,tag=args.rank+10*i + self.count)
                                      
                tensor=tensor.to(device)
                

            return tensor
        
        def close(self):
            self.hook.remove()

    def remove_hooks_params(list_hooks_params):
        if len(list_hooks_params) > 0:
            for hook in list_hooks_params:
                hook.close()
            list_hooks_params.clear()
            print("Removing hooks successfully!!!")



    # load tiny imagenet dataset 
    train_loader = generate_dataloader(args, data=TRAIN_DIR ,name ="train",transform=preprocess_transform)
    test_loader = generate_dataloader(args, data=VALID_DIR ,name ="val",transform=preprocess_transform)

    model=model.to(device)

    hookB = [HookTensor(p,counter) for counter, p in enumerate(model.parameters()) if p.requires_grad]

    optimizer=torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-1)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.3, total_iters=4)

    history = {
    "train_time" : np.zeros(args.epochs + 10000),
    "test_acc" : np.zeros(args.epochs + 10000),
    "train_loss" : np.zeros(args.epochs + 10000),
    "train_acc" : np.zeros(args.epochs + 10000)
    }
    # synchronize workers
    comm.barrier()

    for epoch in range(1, args.epochs + 10000):
        train(comm, args, dograd, model, optimizer, train_loader, test_loader,epoch, history)
        scheduler.step()

    # remove hooks from parameters
    remove_hooks_params(hookB)

    
    
def main():
    parser = argparse.ArgumentParser(description='Distributed training on tiny imagenet dataset')
    parser.add_argument('--rank', type=int, default=0, metavar='R',
                        help='rank for distributed MPI')
    parser.add_argument('--world-size', type=int, default=2, metavar='D',
                        help='partition group (default: 2)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--epochs', type=int, default=20, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--iter-exchange', type=int, default=1, metavar='N',
                        help='how many batches to wait before exchange parameters')
    parser.add_argument('--model-name', type=str, default="resnet18", metavar='N',
                    help='pre-trained model name')
  
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    args.rank = comm.Get_rank()

    args.world_size = comm.Get_size()


    print("rank : ",args.rank)
    print("world size: ",args.world_size)

    # load model
    model = torch.load("pretrained_models/" +  args.model_name +".pth")    

    # data structure
    l=0
    for _,param in model.named_parameters():
        l=l+1
    rows, cols = (l,args.world_size)

    global min_epoch, buff, mygrad, req_s

    # send request status
    req_s=[None for _ in range(cols)]

    # myparams to store store parameters before sending
    mygrad = [None for i in range(rows)]
    mygrad=np.array(mygrad)

    # buff to store received parameters
    buff=[[None for _ in range(rows)] for _ in range(cols)]
    buff=np.array(buff)

    min_epoch=np.zeros((args.world_size))
    # initialize
    l=0
    for param in model.parameters():
        for i in range(args.world_size):
            buff[i][l]=param.clone().detach().cpu().numpy()
        mygrad[l]=param.clone().detach().cpu().numpy()    
        l+=1

    print("mygrad",mygrad.shape)
    print("buf",buff.shape)

    if torch.cuda.device_count() == 1: # single gpu device
        torch.cuda.set_device(0)
    else: # multiple gpu devices
        torch.cuda.set_device(args.rank)
    set_random_seeds(random_seed=42 + args.rank)
    torch.set_num_threads(4)

    worker(comm, args, model)


if __name__ == '__main__':
    main()
