import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
# from apex import amp
# from apex.parallel import DistributedDataParallel as DDP
import argparse
import torch.distributed as dist
from datetime import timedelta


from resnet import ResNet18, ResNet50
from utils import *
from hess import *


def valid(args, net, testLoader, device):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in tqdm(testLoader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    # Training settings
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Total epoch")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="Total epoch")


    # Dataset setting
    parser.add_argument("--dataset_path", type=str, default='../data',
                        help="where is the datset")
    parser.add_argument("--dataset_name", type=str, default='CIFAR10',
                        help="the name of the dataset")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="number of workers of data loader")

    # Regularization setting
    parser.add_argument("--lambda_JR", type=float, default=0.001,
                        help="Coefficieint of Jacobian Regularization")
    parser.add_argument("--Hiter", type=int, default=5,
                        help="Iterations of Hutchinson")
    parser.add_argument("--prob", type=float, default=0.1,
                        help="Random sampling on binomial")
    args = parser.parse_args()


    if args.local_rank==-1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))


    trainLoader, testLoader = get_loader(args)

    net = ResNet18().to(device)


    if args.local_rank!=-1:
        # net = DDP(net, message_size=250000000, gradient_predivide_factor=get_world_size(), delay_allreduce=True)
        net = DDP(net)



    for param in net.parameters():
        param.requires_grad =True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    train_record = []
    test_record = []
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_time = datetime.datetime.now()
    # Train
    correct = 0
    total = 0
    max_test = 0
    for epoch in range(args.epochs):
        net.train()
        hessian_loss = 0.0
        jacobian_loss = 0.0
        running_loss = 0.0
        correct = 0
        total = 0
        trace = 0
        hessian_tr = 0
        for idx, data in tqdm(enumerate(trainLoader, 0)):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            inputs.requires_grad = True

            optimizer.zero_grad()
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss_super = criterion(outputs, labels)

            # Calculate trae with hutchinson
            if idx % 10 == 0:
                trace, hessian_tr = hutchinson(args, net, loss_super, outputs, device)

            loss = loss_super + args.lambda_JR * (trace / args.Hiter)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
            # print statistics
                running_loss += loss_super.item()
                hessian_loss += hessian_tr

        train_acc = correct / total
        # Validation after training
        valid_acc = valid(args, net, testLoader, device)
        if valid_acc >  max_test:
            max_test = valid_acc
        if args.local_rank in [-1, 0]:
            print(f'[Epoch {epoch+1}/{args.epochs}] TRAINING Accuracy : ({(100 * train_acc):3f}%) | TEST Accuracy : ({(100 * valid_acc):3f}%)')
            with open('random_hessian_200.txt','a',encoding='utf-8') as f:
                f.write(f'[Epoch {epoch+1}/{args.epochs}] TRAINING Accuracy : {(100 * train_acc):3f} | TEST Accuracy : {(100 * valid_acc):3f}%')
        scheduler.step()

        train_record.append(train_acc)
        test_record.append(valid_acc)

    if args.local_rank in [-1, 0]:
        print('Finished Training, max test accuracy', 100 * max_test)
        with open('random_hessian_200.txt','a',encoding='utf-8') as f:
            f.write('Best TEST Accuracy inputs: %.3f %% \n' % (100 * max_test))
        end_time = datetime.datetime.now()
        delta = end_time - start_time
        delta_gmtime = time.gmtime(delta.total_seconds())
        duration_str = time.strftime("%H:%M:%S", delta_gmtime)
        print(duration_str)
        plt.plot(train_record,label="train accuracy")
        plt.plot(test_record,label="test accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend()
        plt.savefig('random_hessian_200.png')


if __name__ == "__main__":
    main()

