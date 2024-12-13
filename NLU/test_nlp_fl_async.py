import time
import torch.nn as nn
import argparse
import numpy as np
from utils import *
from mpi4py import MPI
import sys
from torch.utils.data import DataLoader,ConcatDataset
from transformers import BertForSequenceClassification, AdamW
from transformers import AutoTokenizer
from transformers import get_scheduler
from tqdm import tqdm
import os


device = torch.device('cuda')

def train(comm, args, model, optimizer, train_loader, test_loader,epoch, history, scheduler):
    start_time = time.time()
    model.train()
    train_total = 0
    train_total_acc = 0
    train_total_loss = 0
    #loop = tqdm(train_loader, leave=True)
    iter = -1
    for batch in train_loader:  # Removed tqdm wrapper
        iter += 1
        # Move data to the device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # receive average weights among the workers
        for i in range(args.world_size):
            if i!=args.rank:
                while(comm.iprobe(source=i, tag=i+10*args.rank)==True):
                    buff[i]=comm.recv(source=i,tag=i+10*args.rank)
                    new_messages[i]=1
                
                while(comm.iprobe(source=i, tag=100000+i)==True):
                    x = comm.recv(source=i,tag=100000+i)
                    if x == args.epochs:
                        min_epoch[i] = x


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
            
        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step() 

        
        #m_epoch=np.min(min_epoch)

        # send weights to workers
        if iter  % args.iter_exchange == 0:
            l=0
            for param in model.parameters():
                myparams[l]=param.data.clone().detach().cpu()
                l+=1

            
            for i in range(args.world_size):
                if i!=args.rank:
                    if(req_s[i]==None):
                        time.sleep(0.02)
                        req_s[i]=comm.isend(myparams,dest=i,tag=args.rank+10*i)
                    else:
                        v = req_s[i].Test()
                        if(v==True):
                            time.sleep(0.02)
                            req_s[i]=comm.isend(myparams,dest=i,tag=args.rank+10*i)
                    comm.send(epoch,dest=i,tag=100000+args.rank)
            if epoch==args.epochs:
                min_epoch[args.rank]=epoch   

        preds = torch.argmax(logits, dim=1)
        train_total_acc += (preds == labels).sum().item()
        train_total_loss += loss.item()
        train_total += labels.size(0)

        # Update progress bar
        #loop.set_description("Training")
        #loop.set_postfix(loss=loss.item(), accuracy=train_total_acc / train_total)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('[Process-{}]Train Epoch {} <Loss: {:.6f}, Accuracy: {:.2f}%> total time {:3.2f}s'.format(args.rank + 1, epoch, train_total_loss / train_total , 100. * train_total_acc / train_total,elapsed_time))
    history["train_time"][epoch-1] = elapsed_time
    history["train_acc"][epoch-1] = 100. * train_total_acc / train_total
    history["train_loss"][epoch-1] = train_total_loss / train_total

    # Validation
    with torch.no_grad():
        correct = 0
        total = 0
        model.eval()
        for batch in test_loader:  # Removed tqdm wrapper
            # Move data to the device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            preds = torch.argmax(logits, dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            del input_ids, labels, logits

        history["test_acc"][epoch - 1] = correct / total
        print('[INFOS-Process_{}] - Accuracy of the network on the {} validation images: {:.2f} %'.format(args.rank + 1, len(test_loader)*args.batch_size_eval, 100 * correct / total)) 
        sys.stdout.flush()   

    if(all_elements_equal(min_epoch, args.epochs)):
        comm.barrier()
        print("[INFOS-Process_{}] - Total training time: {:.2f}" .format(args.rank + 1, sum(history["train_time"])))
        torch.save(history, './log_nlp/' + 'FL_ASYNC__world_' + str(args.world_size) + "_" + args.dataset_name +  '_worker_' + str( args.rank + 1) + '_' + args.model_name + '_history.pt')
        sys.exit(0)



def worker(comm, args, model,encoded_dataset):
    print(encoded_dataset)
    train_loader = generate_dataloader_nlp(args, name ="train", dataset=encoded_dataset["train"])
    if "validation" in encoded_dataset: #MNLI Task
        test_loader = generate_dataloader_nlp(args, name ="val", dataset=encoded_dataset["validation"])
    elif "validation_matched" in encoded_dataset:
        validation_dataset = ConcatDataset([encoded_dataset["validation_matched"], encoded_dataset["validation_mismatched"]])
        test_loader = generate_dataloader_nlp(args, name ="val", dataset=validation_dataset)
    else:
        test_loader = generate_dataloader_nlp(args, name ="val", dataset=encoded_dataset["test"])

    model=model.to(device)

    # Define optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    num_training_steps = len(train_loader) * args.epochs 
    scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)


    history = {
    "train_time" : np.zeros(args.epochs + 10000),
    "test_acc" : np.zeros(args.epochs + 10000),
    "train_loss" : np.zeros(args.epochs + 10000),
    "train_acc" : np.zeros(args.epochs + 10000)
    }
    # synchronize workers
    comm.barrier()

    for epoch in range(1, args.epochs + 10000):
        train(comm, args, model, optimizer, train_loader, test_loader,epoch, history, scheduler)
        torch.save(history, './log_nlp/' + 'FL_ASYNC__world_' + str(args.world_size) + "_" + args.dataset_name +  '_worker_' + str( args.rank + 1) + '_' + args.model_name + '_history.pt')
    

    
    
def main(encoded_dataset, num_labels, dataset_name):
    parser = argparse.ArgumentParser(description='Distributed training on tiny imagenet dataset')
    parser.add_argument('--rank', type=int, default=0, metavar='R',
                        help='rank for distributed MPI')
    parser.add_argument('--world-size', type=int, default=2, metavar='D',
                        help='partition group (default: 2)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--batch-size-eval', type=int, default=64, metavar='N',
                            help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--iter-exchange', type=int, default=1, metavar='N',
                        help='how many batches to wait before exchange parameters')
    parser.add_argument('--model-name', type=str, default="bert-base-uncased", metavar='N',
                    help='pre-trained model name')  
    parser.add_argument('--dataset-name', type=str, default="sst2", metavar='N',
                    help='dataset name')  
  
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    args.rank = comm.Get_rank()

    args.world_size = comm.Get_size()
    args.dataset_name = dataset_name

    print("rank : ",args.rank)
    print("world size: ",args.world_size)

    # load model
    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)


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

    worker(comm, args, model, encoded_dataset)


if __name__ == '__main__':
    glue_benchmark, datasets_num_labels = glue_data()
    for task_name, num_labels in datasets_num_labels.items():
        # get dataset
        print("Loading dataset {}".format(task_name))
        dataset = glue_benchmark[task_name]
        # Apply the tokenizer to the dataset
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        def preprocess_glue_task(examples, task_name):
            task_fields = {
                "mnli": ("premise", "hypothesis"),
                "qqp": ("question1", "question2"),
                "qnli": ("question", "sentence"),
                "sst2": ("sentence", None),
                "cola": ("sentence", None),
                "stsb": ("sentence1", "sentence2"),
                "mrpc": ("sentence1", "sentence2"),
                "rte": ("sentence1", "sentence2")
            }
            
            field1, field2 = task_fields[task_name]
            
            if field2 is None:
                return tokenizer(
                    examples[field1], 
                    truncation=True, 
                    padding="max_length", 
                    max_length=128
                )
            else:
                return tokenizer(
                    examples[field1], 
                    examples[field2], 
                    truncation=True, 
                    padding="max_length", 
                    max_length=128
                )
                    
        encoded_dataset = dataset.map(lambda examples: preprocess_glue_task(examples, task_name),  batched=True, keep_in_memory=True)

        # Convert dataset to PyTorch tensors
        encoded_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        
        main(encoded_dataset, num_labels, task_name)

