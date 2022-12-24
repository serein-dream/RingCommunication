import os
import argparse
import datetime
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data as tud
import utils
import models
from torchvision import datasets, transforms
import torchvision
import torch.nn as nn

#Set parameters
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="./data/", help='Dataset path.')
    parser.add_argument('--dataset', type=str, default="mnist", help='Dataset type.')
    parser.add_argument('--device', type=str, default="cpu", help='Dataset type.')
    parser.add_argument('--world_size', type=int, default=4, help="World size.")
    parser.add_argument('--model', type=str, default="cnn", help="Choose your model.")
    #parser.add_argument("--hidden_layers", nargs='+', type=int, default=[64, 32, 16])
    #parser.add_argument("--dropouts", nargs='+', type=float, default=[0.5, 0.5])
    parser.add_argument("--method", choices=['multiclass', 'binary', 'regression'], default='multicalss', type=str)
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size.')
    parser.add_argument('--lr', type=float, default=0.1, help='Training learning rate.')
    parser.add_argument('--lr_gamma', type=float, default=0.9, help='Training learning rate gamma.')
    parser.add_argument('--lr_step', type=int, default=[40, 80], nargs="+", help='Training learning rate steps.')
    parser.add_argument('--epoch', type=int, default=200, help='Training epochs.')
    parser.add_argument('--backend', type=str, default="gloo", help='Communication backend.')

    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    args = parser.parse_args()

    return args, time


def init_process(rank, args):
    #Parameter configuration and initialization
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ["MASTER_PORT"] = "5555"  
    #print("rank ",rank,"starts\n")
    dist.init_process_group(args.backend, rank=rank, world_size=args.world_size)
    device = torch.device("cpu")
    if args.device != "cpu":
        device = torch.device("cuda", dist.get_rank())
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1，2"
    torch.manual_seed(1111)

    #Prepare mnist dataset for cnn
    train_set=datasets.MNIST(args.data_dir+args.dataset,train=True, transform=torchvision.transforms.ToTensor(), download=True)
    test_set=datasets.MNIST(args.data_dir+args.dataset,train=False, transform=torchvision.transforms.ToTensor(), download=True)

    sampler = tud.distributed.DistributedSampler(dataset=train_set, shuffle=True)
    train_loader = tud.DataLoader(dataset=train_set, batch_size=args.batch_size, sampler=sampler)
    test_loader = tud.DataLoader(dataset=test_set, batch_size=args.batch_size, shuffle=True)

    #Cnn related components
    if args.model == 'cnn':
        model = models.CNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.RMSprop(model.parameters( ), lr=args.lr, alpha=args.lr_gamma)#optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)      
    else:
        exit('Error: unrecognized model')
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log = utils.Log()
    log.get_logger()
    log.add_handler(time)
    log.logger.info(f"Start time: {time}")
    log.logger.info(f"rank = {rank} is initialized")
    log.logger.info(f"Set device {dist.get_rank()}")

    model.to(device)
    #The following function calls enter the iterative training of cnn
    utils.run(model, train_loader, test_loader, sampler, optimizer, criterion,
              args.lr, args.lr_gamma, args.lr_step, args.epoch, log, device)   
    
    
    if rank == 0:
        log.save_model(model)
    log.save_metric()
    

#Program entry is here
if __name__ == "__main__":
    args, time = get_args()
    process = []
    mp.set_start_method("spawn")
    for rank in range(args.world_size):
        p = mp.Process(target=init_process, args=(rank, args))
        p.start()
        process.append(p)

    for p in process:
        p.join()
