import time
import torch.nn as nn
import argparse
import numpy as np
from utils import *
from mpi4py import MPI
import sys

device = torch.device('cuda')

# loss function
criterion = nn.CrossEntropyLoss()


def train(comm, args, model, optimizer, train_loader, test_loader,epoch, history):
    start_time = time.time()
    model.train()
    train_total = 0
    train_total_acc = 0
    train_total_loss = 0
    for iter, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # receive average weights among the workers
        for i in range(args.world_size):
            if i!=args.rank:
                while(comm.iprobe(source=i, tag=i+10*args.rank)==True):
                    buff[i]=comm.recv(source=i,tag=i+10*args.rank)
                    new_messages[i]=1
          
                while(comm.iprobe(source=i, tag=100000+i)==True):
                    min_epoch[i]=comm.recv(source=i,tag=100000+i)


        if (np.sum(new_messages) >= 1):
            l=0
            for param in model.parameters():
                param.data = param.data.detach().cpu()
                val=param.data.clone()
                for i in range(args.world_size):
                    if(i!=args.rank):
                        val+=buff[i][l]       
                        
                param.data=val/args.world_size
                param.data=param.data.to(device)
                
                         
                l+=1
            for i in range(args.world_size):
                new_messages[i]=0
            
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step() 

        m_epoch=np.min(min_epoch)

        # send weights to workers
        if iter  % args.iter_exchange == 0:
            l=0
            for param in model.parameters():
                myparams[l]=param.data.clone().detach().cpu()
                l+=1

            
            for i in range(args.world_size):
                if i!=args.rank:
                    if(req_s[i]==None):
                        time.sleep(0.01)
                        req_s[i]=comm.isend(myparams,dest=i,tag=args.rank+10*i)
                    else:
                        v = req_s[i].Test()
                        if(v==True):
                            time.sleep(0.01)
                            req_s[i]=comm.isend(myparams,dest=i,tag=args.rank+10*i)
                    comm.send(epoch,dest=i,tag=100000+args.rank)
            min_epoch[args.rank]=epoch   

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
        torch.save(history, './log/' + 'FL_ASYNC__world_' + str(args.world_size) + '_worker_' + str( args.rank + 1) + '_' + args.model_name + '_history.pt')
        sys.exit(0)


def worker(comm, args, model):
    # load tiny imagenet dataset 
    train_loader = generate_dataloader(args, data=TRAIN_DIR ,name ="train",transform=preprocess_transform)
    test_loader = generate_dataloader(args, data=VALID_DIR ,name ="val",transform=preprocess_transform)

    model=model.to(device)

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
        train(comm, args, model, optimizer, train_loader, test_loader,epoch, history)
        scheduler.step()

    
    
def main():
    parser = argparse.ArgumentParser(description='Distributed training on tiny imagenet dataset')
    parser.add_argument('--rank', type=int, default=0, metavar='R',
                        help='rank for distributed MPI')
    parser.add_argument('--world-size', type=int, default=2, metavar='D',
                        help='partition group (default: 2)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
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

    global min_epoch, myparams, buff,req_s, new_messages

    # send request status
    req_s=[None for _ in range(cols)]

    # myparams to store store parameters before sending
    myparams = [None for i in range(rows)]
    myparams=np.array(myparams)    

    # buff to store received parameters
    buff=[[None for _ in range(rows)] for _ in range(cols)]
    buff=np.array(buff)

    min_epoch=np.zeros((args.world_size))
    new_messages=np.zeros((args.world_size))


    # initialize
    l=0
    for param in model.parameters():
        for i in range(args.world_size):
            buff[i][l]=param.data.clone().detach().cpu().numpy() 
        myparams[l]=param.data.clone().detach().cpu().numpy()  
        l+=1

    print("buf",buff.shape)
    print("buf",myparams.shape)

    if torch.cuda.device_count() == 1: # single gpu device
        torch.cuda.set_device(0)
    else: # multiple gpu devices
        torch.cuda.set_device(args.rank)

    set_random_seeds(random_seed=42 + args.rank)
    torch.set_num_threads(4)

    worker(comm, args, model)


if __name__ == '__main__':
    main()