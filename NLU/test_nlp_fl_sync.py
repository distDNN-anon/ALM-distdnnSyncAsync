import time
import torch.nn as nn
import argparse
import numpy as np
from utils import *
from mpi4py import MPI
import sys
from torch.utils.data import DataLoader, ConcatDataset
from transformers import BertForSequenceClassification, AdamW
from transformers import AutoTokenizer
from transformers import get_scheduler
from tqdm import tqdm
import os



device = torch.device('cuda')
d_cpu = torch.device('cpu')


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

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # allReduice weights
        if iter  % args.iter_exchange == 0:
            for param in model.parameters():
                param.data=param.data.to(d_cpu)
                buf_grad = param.data.clone()
                time.sleep(0.02)
                comm.Allreduce(param.data, buf_grad, op=MPI.SUM)
                param.data=buf_grad.clone()
                param.data=param.data.to(device)
                param.data= param.data.div_(args.world_size)           

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

    #if args.rank == 0:
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


def worker(comm, args, model, encoded_dataset):
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
    "train_time" : np.zeros(args.epochs),
    "test_acc" : np.zeros(args.epochs),
    "train_loss" : np.zeros(args.epochs),
    "train_acc" : np.zeros(args.epochs)
    }
    # synchronize workers
    comm.barrier()

    for epoch in range(1, args.epochs + 1):
        train(comm, args, model, optimizer, train_loader, test_loader,epoch, history,scheduler)
        torch.save(history, './log_nlp/' + 'FL_SYNC_world_' + str(args.world_size) + "_" + args.dataset_name + '_worker_'  + str( args.rank + 1) + '_' + args.model_name + '_history.pt')

    comm.barrier()  
    #if args.rank==0:
    print("[INFOS-Rank_{}] - Total training time: {:.2f}" .format(args.rank + 1, sum(history["train_time"])))
    torch.save(history, './log_nlp/' + 'FL_SYNC_world_' + str(args.world_size) + "_" + args.dataset_name + '_worker_'  + str( args.rank + 1) + '_' + args.model_name + '_history.pt')

    
    
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
    parser.add_argument('--iter_exchange', type=int, default=1, metavar='N',
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
        if task_name == "mnli":
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
           
