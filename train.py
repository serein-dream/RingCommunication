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
import socketserver

#Set parameters
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="null", help='Dataset path.')
    parser.add_argument('--data_dir', type=str, default="./data/", help='Dataset path.')
    parser.add_argument('--dataset', type=str, default="mnist", help='Dataset type.')
    parser.add_argument('--device', type=str, default="cpu", help='Dataset type.')
    parser.add_argument('--model', type=str, default="cnn", help="Choose your model.")
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size.')
    parser.add_argument('--lr', type=float, default=0.01, help='Training learning rate.')
    parser.add_argument('--lr_gamma', type=float, default=0.9, help='Training learning rate gamma.')
    #parser.add_argument('--lr_step', type=int, default=[40, 80], nargs="+", help='Training learning rate steps.') define in run.py
    parser.add_argument('--epoch', type=int, default=200, help='Training epochs.')
    parser.add_argument('--backend', type=str, default="gloo", help='Communication backend.')
    parser.add_argument(
        '-i',
        '--init-method',
        type=str,
        default='tcp://10.254.19.19:23456',
        help='URL specifying how to initialize the package.')
    parser.add_argument('--world-size', type=int, default=7, help='Number of processes participating in the job.')
    parser.add_argument('--rank', type=int, default=0, help='the rank of current process in all process.')
    parser.add_argument('--gloo_socket_ifname', type=str, default="enp6s0", help='you can get it by "ifconfig" if you meet connection refused')
    args = parser.parse_args()

    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_path = os.path.join("Log", time)
    if os.path.exists(log_path) == False:
        os.makedirs(log_path)
    parameter_path = os.path.join(log_path, "parameter.txt")

    with open(parameter_path, 'w') as f:
        sys.stdout = f
        for arg in vars(args):
            print(format(arg, '<30'), format(str(getattr(args, arg)), '<'))

    return args, time


def init_process(time,rank, args):
    #Parameter configuration and initialization
    os.environ["GLOO_SOCKET_IFNAME"]=args.gloo_socket_ifname
    dist.init_process_group(backend=args.backend, init_method=args.init_method, world_size=args.world_size, rank=args.rank)
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
        optimizer = torch.optim.RMSprop(model.parameters( ), lr=args.lr, alpha=args.lr_gamma)
    else:
        exit('Error: unrecognized model')

    if args.model_path != "null":
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    log = utils.Log()
    log.get_logger()
    log.add_handler(time)
    log.logger.info(f"Initialization is done,rank = {rank}")

    model.to(device)
    #The following function calls enter the iterative training of cnn
    utils.run(model, train_loader, test_loader, sampler, optimizer, criterion,
              args.lr, args.lr_gamma, args.epoch, log, device)   
    
    
    log.save_model(model)
    log.save_metric()
    

#Program entry is here
if __name__ == "__main__":
    args, time = get_args()
    socketserver.TCPServer.allow_reuse_address = True
    init_process(time,args.rank,args)

