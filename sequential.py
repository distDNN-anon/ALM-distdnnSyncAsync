import time
import torch.nn as nn
import argparse
import numpy as np
from utils import *
from mpi4py import MPI

device = torch.device('cuda')

# loss function
criterion = nn.CrossEntropyLoss()


def train(args, model, optimizer, train_loader, epoch, train_loss_log, train_acc_log,train_time_log):
    start_time = time.time()
    model.train()
    train_total = 0
    train_total_acc = 0
    train_total_loss = 0
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step() 
        i += 1
        train_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
        train_correct = train_pred.eq(target.view_as(train_pred)).sum().item()
        train_total_acc +=  train_correct
        train_total_loss += loss.item()
        train_total += target.shape[0]
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('[Process-{}]Train Epoch {} <Loss: {:.6f}, Accuracy: {:.2f}%> total time {:3.2f}s'.format(args.rank + 1, epoch, train_total_loss / train_total , 100. * train_total_acc / train_total,elapsed_time))
    train_time_log[epoch-1] = elapsed_time
    train_acc_log[epoch-1] = 100. * train_total_acc / train_total
    train_loss_log[epoch-1] = train_total_loss / train_total 


def test(args, model, test_loader, epoch, test_loss_log, test_acc_log):
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            test_pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            test_correct += test_pred.eq(target.view_as(test_pred)).sum().item()
            test_total += target.shape[0]
        test_acc = float(test_correct) / float(test_total)
        test_loss /= float(test_total)
    print("[Process_{}]-Epoch {} Test Loss: {:.6f}; Test Accuracy: {:.2f}%.\n".format(args.rank + 1, epoch, test_loss, 100 * test_acc))
    test_loss_log[epoch - 1] = test_loss
    test_acc_log[epoch - 1] = test_acc



def worker(args, model):
    # load tiny imagenet dataset 
    train_loader = generate_dataloader(args, data=TRAIN_DIR ,name ="train",transform=preprocess_transform)
    test_loader = generate_dataloader(args, data=VALID_DIR ,name ="val",transform=preprocess_transform)

    model=model.to(device)


    optimizer=torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=5e-1)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.3, total_iters=4)

    epochs = args.epochs
    train_time_log = np.zeros(epochs)
    test_loss_log = np.zeros(epochs)
    test_acc_log = np.zeros(epochs)
    train_loss_log = np.zeros(epochs)
    train_acc_log = np.zeros(epochs)    
    for epoch in range(1, epochs + 1):
        train(args, model, optimizer, train_loader, epoch, train_loss_log, train_acc_log,train_time_log)
        test(args, model, test_loader, epoch, test_loss_log, test_acc_log)
        scheduler.step()
        

    print("Total training time: {:.2f}" .format(sum(train_time_log)))
    print('\ntest model...')
    test(args, model, test_loader, epochs, test_loss_log, test_acc_log)
    np.savetxt('./log/' + args.model_name + 'seq_train_time.log', train_time_log, fmt='%1.4f', newline=' ')
    np.savetxt('./log/' + args.model_name + 'seq_train_acc.log', train_acc_log, fmt='%1.4f', newline=' ')
    np.savetxt('./log/' + args.model_name + 'seq_train_loss.log', train_loss_log, fmt='%1.4f', newline=' ')      
    np.savetxt('./log/' + args.model_name + 'seq_test_loss.log', test_loss_log, fmt='%1.4f', newline=' ')
    np.savetxt('./log/' + args.model_name + 'seq_test_acc.log', test_acc_log, fmt='%1.4f', newline=' ')
    
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

    print("rank",args.rank)
    print("world",args.world_size)

    # load model
    model = torch.load("pretrained_models/" +  args.model_name +".pth")    

    
    if torch.cuda.device_count() == 1: # single gpu device
        torch.cuda.set_device(0)
    else: # multiple gpu devices
        torch.cuda.set_device(args.rank)

    set_random_seeds(random_seed=42 + args.rank)
    torch.set_num_threads(4)

    worker(args, model)


if __name__ == '__main__':
    main()
