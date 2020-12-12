import os
import shutil
import time
import argparse
import copy
import numpy as np
from math import ceil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.distributed as dist
from models import *
import random
torch.multiprocessing.set_sharing_strategy('file_system')
parser = argparse.ArgumentParser(description='PyTorch Cifar10 Training')
parser.add_argument('-epochs', default=160, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-lr', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('-rank', default=0, type = int,
                    help='rank of current process')
parser.add_argument('-world_size', default=3, type = int,
                    help="world size")
parser.add_argument('-init_method', default='tcp://127.0.0.1:23412',
                    help="init-method")
args = parser.parse_args()
graph = get_graph(args.world_size-1, 'ring')
epsilon = 1e-1
filename = "n4epsilon1e-1stationary.txt"
def false_W(beta):
    W = torch.zeros([args.world_size-1,args.world_size-1])
    for i in range(args.world_size-1):
        for j in range(args.world_size-1):
            if(j==i):
                W[i,j] = 0.25
            elif(j in graph[i]):
                W[i,j] = 0.25
            else:
                W[i,j] = 0.25
    return W
def generate_graph(beta):
    #对于4个节点，从0开始到3
    W = torch.zeros([args.world_size-1,args.world_size-1])
    dynamic_graph = copy.deepcopy(graph)
    new_edges = ceil(beta*(args.world_size-1))
    for i in range(new_edges):
        u,v = random.sample(range(args.world_size-1),2)
        if((v not in dynamic_graph[u]) and u!=v):
            dynamic_graph[u].add(v)
            dynamic_graph[v].add(u)
    for i in range(args.world_size-1):
        sumj = 0
        for j in range(args.world_size-1):
            if(j!=i):
                if(j in dynamic_graph[i]):
                    W[i,j] = 1/max(len(dynamic_graph[i]),len(dynamic_graph[j]))
                    sumj+=W[i,j]
                else:
                    W[i,j] = 0
        W[i,i] = 1-sumj
    return W
def generate_dynamic():
    W = torch.zeros([args.world_size-1,args.world_size-1])
    dynamic_graph = copy.deepcopy(graph)
    flag = np.random.randint(2)
    if(flag):
        v = np.random.randint(args.world_size-1)
        u = (v+1)%(args.world_size-1)
        dynamic_graph[u].remove(v)
        dynamic_graph[v].remove(u)
    else:
        v = np.random.randint(args.world_size-1)
        u = (v+2)%(args.world_size-1)
        dynamic_graph[v].add(u)
        dynamic_graph[u].add(v)
    for i in range(args.world_size-1):
        sumj = 0
        for j in range(args.world_size-1):
            if(j!=i):
                if(j in dynamic_graph[i]):
                    W[i,j] = 1/max(len(dynamic_graph[i]),len(dynamic_graph[j]))
                    sumj+=W[i,j]
                else:
                    W[i,j] = 0
        W[i,i] = 1-sumj
    return W
def stationay_graph():
    W = torch.zeros([args.world_size-1,args.world_size-1])
    dynamic_graph = copy.deepcopy(graph)
    for i in range(args.world_size-1):
        sumj = 0
        for j in range(args.world_size-1):
            if(j!=i):
                if(j in dynamic_graph[i]):
                    W[i,j] = 1/max(len(dynamic_graph[i]),len(dynamic_graph[j]))
                    sumj+=W[i,j]
                else:
                    W[i,j] = 0
        W[i,i] = 1-sumj
    return W
def coordinate(rank, world_size):
    output = open(filename, "w")
    #args = parser.parse_args()
    model = resnet20()
    model = model.cpu()
    model_flat = flatten_all(model)
    dist.broadcast(model_flat, world_size)
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cpu()
    #cudnn.benchmark = True

    # Data loading code
    train_transform = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    val_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    trainset = datasets.CIFAR10(root='./data', train=True,download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=128,pin_memory=False,shuffle=False, num_workers=2)

    #valset = datasets.CIFAR10(root='./data', train=False,download=False, transform=val_transform)
    #val_loader = torch.utils.data.DataLoader(valset, batch_size=100,pin_memory=False,shuffle=False, num_workers=2)
   
    time_cost = 0
    for epoch in range(args.epochs):

        print(epoch, "in total", args.epochs)
        #print('masterFirst')
        dist.barrier()
        W = stationay_graph()#beta
        dia = 0
        for i in range(world_size):
            if W[i,i]>1e-3:
                dia+=1
        num_of_edge = (sum(W[W>1e-3])-dia)//2
        print(W,num_of_edge)
        
        dist.broadcast(W, world_size)
        #print('masterSecond')
        t1 = time.time()
        dist.barrier()
        #print('masterTTTTT')
        #dist.barrier()
        t2 = time.time() 
        time_cost += t2 - t1
        model_flat.zero_()
        loss = torch.FloatTensor([0])
        dist.reduce(loss, world_size, op=dist.reduce_op.SUM)
        loss.div_(world_size)
        dist.reduce(model_flat, world_size, op=dist.reduce_op.SUM)
        model_flat.div_(world_size)
        print("reduce OK in epoch", epoch,"avg model is",torch.mean(model_flat))
        
        unflatten_all(model, model_flat)
        # evaluate on validation set
        #_ ,prec1 = validate(val_loader, model, criterion)
        print('loss:',loss.item())
        output.write('%d %3f %3f %d\n'%(epoch,time_cost,loss.item(),num_of_edge))
        output.flush()
    
    output.close()

def run(rank, world_size):
    print('Start node: %d  Total: %3d'%(rank,world_size))
    #args = parser.parse_args()
    current_lr = args.lr
    adjust = [80,120]


    model = resnet20()
    model = model.cpu()
    model_flat = flatten_all(model)
    dist.broadcast(model_flat, world_size)
    unflatten_all(model, model_flat)
    #model_l = flatten(model)
    #model_r = flatten(model)
    model_recv = []
    for i in range(world_size-1):
        model_recv.append(flatten(model))
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cpu()

    optimizer = torch.optim.SGD(model.parameters(), lr=current_lr, weight_decay=0.0001)

    #cudnn.benchmark = True

    # Data loading code
    train_transform = transforms.Compose(
        [transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    trainset = datasets.CIFAR10(root='./data', train=True,download=True, transform=train_transform)
    sz = trainset.__len__()
    data_to_use = 2048
    trainset, _com = torch.utils.data.random_split(trainset,[data_to_use,sz-data_to_use])
    train_sampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=data_to_use//(1*world_size),pin_memory=False,shuffle=False, num_workers=2, sampler=train_sampler)
    
    val_transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    valset = datasets.CIFAR10(root='./data', train=False,download=False, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=100,pin_memory=False,shuffle=False, num_workers=4)
  
    for epoch in range(args.epochs):
        #print("rank:", rank, "in epoch", epoch)
        dist.barrier()
        W = torch.zeros([args.world_size-1,args.world_size-1])
        dist.broadcast(W, world_size)
        if epoch in adjust: 
            current_lr = current_lr * 0.1    
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

       
        train_sampler.set_epoch(epoch)	
        loss = train(train_loader, model, criterion, optimizer, epoch, rank, world_size, model_recv,W,current_lr)
        #print('lolala')
        dist.barrier()
        #print('lolalaAfter')
        model_flat = flatten_all(model)
        
        dist.reduce(torch.FloatTensor([loss]), world_size, op=dist.reduce_op.SUM)
        dist.reduce(model_flat, world_size, op=dist.reduce_op.SUM)

        #output.write('Epoch: %d  Time: %3f  Train_loss: %3f  Val_acc: %3f\n'%(epoch,time_cost,loss,prec1))

def train(train_loader, model, criterion, optimizer, epoch, rank, world_size, model_recv,W,current_lr):
    losses = AverageMeter()
    top1 = AverageMeter()

   
    model.train()
    #print('loader:',len(list(enumerate(train_loader))))
    for i, (input, target) in enumerate(train_loader):
        #print('rank:', rank, 'batch:', i)
        #if i>0:
        #    break
        input_var = torch.autograd.Variable(input.cpu())
        #target = target.cpu(async=True)
        target_var = torch.autograd.Variable(target)
      
        output = model(input_var)
        loss = criterion(output, target_var)

       
        optimizer.zero_grad()
        loss.backward()
        #???
        for param in model.parameters():
            param.grad.data.add_(0.0001,param.data)   
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        optimizer.step()
        # communicate
        model_flat = flatten(model)
        print("parameter of rank",rank,torch.mean(model_flat)*1000000)
        grad_flat = flatten_grad(model)
        print("gradient of rank",rank,torch.mean(grad_flat)*1000000)
        #print(rank, model_flat.mean())
        #print("shape", model_flat.shape)
        global epsilon
        noise = torch.tensor(np.random.laplace(0,2*torch.abs(torch.mean(grad_flat))*current_lr/epsilon,model_flat.shape))
        model_flat.add_(noise)
        broadcast(model_flat, rank, world_size, model_recv,W)
        #dist.barrier()
        #model_flat.add_(model_l)
        #model_flat.add_(model_r)
        model_flat.mul_(W[rank,rank])
        for neighbor_model in model_recv:
            #print('rank+nei', rank, neighbor_model.mean())
            model_flat.add_(neighbor_model)
        #model_flat.div_(len(graph[rank])+1)
        #print(rank,'after',model_flat.mean())
        print("After avg of rank",rank,torch.mean(model_flat)*1000000)
        unflatten(model, model_flat)
        #optimizer.step()

    return losses.avg


def validate(val_loader, model, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        #print('validate', i)
        input_var = torch.autograd.Variable(input.cpu())
        #target = target.cuda(async=True)
        target_var = torch.autograd.Variable(target)

        # compute output
        with torch.no_grad():
            output = model(input_var)
            loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))
        losses.update(loss.data.item(), input.size(0))
        top1.update(prec1[0], input.size(0))

    return losses.avg, top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def broadcast(data, rank, world_size, recv_buff,W):
    cnt = 0
    recv_req = []
    neighbors = [x for x in range(world_size) if x != rank]
    for neighbor in neighbors:
        req = dist.irecv(recv_buff[cnt], src = neighbor)
        recv_req.append(req)
        #print("rank", rank, "ready to receive data from", neighbor)
        cnt+=1
    #dist.barrier()
    for neighbor in neighbors:
        #print(cnt)
        req = dist.isend(data.mul(W[neighbor,rank]), dst = neighbor)
        req.wait()
        #print("rank", rank, "send data to", neighbor)
    for req in recv_req:
        req.wait()
        #dist.recv(recv_buff[cnt], src = neighbor)
        #print("rank", rank, "ready to receive data from", neighbor)

def flatten_all(model):
    vec = []
    for param in model.parameters():
        vec.append(param.data.view(-1))
    #for b in model._all_buffers():
    #    vec.append(b.data.view(-1))
    return torch.cat(vec)

def unflatten_all(model, vec):
    pointer = 0
    for param in model.parameters():
        num_param = torch.prod(torch.LongTensor(list(param.size())))
        param.data = vec[pointer:pointer + num_param].view(param.size())
        pointer += num_param
    #for b in model._all_buffers():
    #    num_param = torch.prod(torch.LongTensor(list(b.size())))
    #    b.data = vec[pointer:pointer + num_param].view(b.size())
    #    pointer += num_param

def flatten(model):
    vec = []
    for param in model.parameters():
        vec.append(param.data.view(-1))
    return torch.cat(vec)
def flatten_grad(model):
    vec = []
    for param in model.parameters():
        vec.append(param.grad.data.view(-1))
    return torch.cat(vec)

def unflatten(model, vec):
    pointer = 0
    for param in model.parameters():
        num_param = torch.prod(torch.LongTensor(list(param.size())))
        param.data = vec[pointer:pointer + num_param].view(param.size())
        pointer += num_param

if __name__ == '__main__':
    #dist.init_process_group('mpi')
    #W = np.zeros([args.word_size-1][args.word_size-1])
    dist.init_process_group(backend="gloo", init_method = args.init_method, world_size= args.world_size, rank=args.rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if rank == world_size - 1:
        coordinate(rank, world_size - 1)
    else:
        run(rank, world_size - 1)
