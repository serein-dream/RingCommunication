import time
import torch
import torch.distributed as dist
from sklearn.metrics import accuracy_score
import numpy as np
import torch.nn.functional as F
import socket

#change lr, according to lr_ Gamma
def adjust_learning_rate(epoch, lr, lr_gamma, optimizer, log):
    if epoch%25==0:
        if lr<0.008 and lr>0.005:
            lr_gamma=0.93
        if lr<=0.005:
            lr_gamma=0.97
        lr = lr * lr_gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        log.logger.warning(f"Current learning rate: {lr}")
    return lr


def train_model(model, rank, train_loader, optimizer, criterion,
                epoch, max_epoch, log, device):
    model.train()
    gather_loss=0
    correct=0
    total=0
    accuracy=0

    for batch_idx,data in enumerate(train_loader,0):
        inputs,target = data
        inputs=inputs/255.0
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        size = int(dist.get_world_size())
        rank = dist.get_rank()
        #com_start_time=time.time()
        for name, param in model.named_parameters():
            left_grad= param.grad.data
            #begin from rank 0
            if rank ==0:
                dist.send(param.grad.data,dst=(rank+1)%size,group=None, tag=0)
               
            else:
            #others, firstly, recive its right worker; secondely, add its grad and updata itself; thirdly, send the new grad to its right worker
                dist.recv(left_grad,src=(rank+size-1)%size,group=None, tag=0)             
                param.grad.data+=left_grad
                dist.send(param.grad.data,dst=(rank+1)%size,group=None, tag=0)

            #after all workers done ,the last rank aggregate all worker's grad ,send to rank 0

            #Now spread the aggregated grad ring one-to-one
            #The grad of rank size-1 is the grad that aggregates all the rank, so rank size-2 does not need to be sent to rank size-1 again
            if rank < size-1:   
                dist.recv(param.grad.data,src=(rank-1)%size,group=None, tag=0)
                dist.send(param.grad.data,dst=(rank+1)%size,group=None, tag=0)
                
            else:
                dist.recv(param.grad.data,src=(rank-1)%size,group=None, tag=0)
                
            #average       
            param.grad.data /= size
        #com_end_time=time.time()
        #log.logger.warning(f"Rank: {rank}, Epoch: [{epoch + 1}/{max_epoch}].time of communication: {com_end_time-com_start_time}. ")
        optimizer.step()
        gather_loss+=loss.item()*target.size(0)   
        _,predicted = torch.max(outputs,dim=1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        accuracy=correct / total
        train_loss = gather_loss / total

    log.add_metric("Train_loss", train_loss)
    log.add_metric("Train_accuracy", accuracy)  
    log.logger.warning(
        f"Rank: {rank}, Epoch: [{epoch + 1}/{max_epoch}]. Train Loss: {train_loss}, Train_accuracy: {accuracy * 100}%.")
    return accuracy


@torch.no_grad()
def test_model(model, rank, test_loader, criterion, epoch, max_epoch, log, device):
    model.eval()
    correct=0
    total=0
    accuracy=0

    for data in test_loader:
        inputs,target = data
        inputs=inputs/255.0
        inputs, target = inputs.to(device), target.to(device)
        outputs = model(inputs)
        _,predicted = torch.max(outputs.data,dim=1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        accuracy=correct / total
    log.add_metric("Test_accuracy", accuracy)
    log.logger.warning(
        f"Rank: {rank}, Epoch: [{epoch + 1}/{max_epoch}].Test_accuracy: {accuracy * 100}%.")
    return accuracy


def run(model, training_loader, test_loader, sampler, optimizer, criterion,
        lr, lr_gamma, epochs, log, device):
    rank = dist.get_rank()
    best_train_acc = -1
    best_test_acc = -1
    begin_train_time=time.time()
    for epoch in range(epochs):
        start_time = time.time()
        sampler.set_epoch(epoch)
        lr = adjust_learning_rate(epoch, lr, lr_gamma, optimizer, log)
        train_acc = train_model(model, rank, training_loader, optimizer, criterion, epoch,
                                epochs, log, device)       
        best_train_acc = max(best_train_acc, train_acc)
        test_acc = test_model(model, rank, test_loader, criterion, epoch, epochs, log, device)
        end_time=time.time()
        log.logger.warning(f"Rank: {rank}, Epoch: [{epoch + 1}/{epochs}].time of this epoch: {end_time-start_time}.   and totle time: {end_time-begin_train_time}")
        best_test_acc = max(best_test_acc, test_acc)
        '''
        if rank == 0:
            test_acc = test_model(model, rank, test_loader, criterion, epoch, epochs, log, device)
            best_test_acc = max(best_test_acc, test_acc)
        '''
        if epoch>=300 and epoch%50==0:
            log.save_model_e(model,epoch)

    log.logger.warning(f"Best train accuracy: {best_train_acc}")
    if rank == 0:
        log.logger.warning(f"Best test accuracy: {best_test_acc}")

