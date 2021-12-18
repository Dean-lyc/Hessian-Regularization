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
import os


from models.resnet import ResNet18, ResNet50, Wide_ResNet28_10
from models.resnet_imagenet import resnet18, resnet50
from utils import *
from hess import *


def valid(args, net, testLoader, device):
    net.eval()
    if args.dataset_name == "cifar10":
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

    elif args.dataset_name == "cifar100" or "imagenet":
        correct = 0
        total = 0
        correct_1 = 0
        correct_5 = 0
        with torch.no_grad():
            for data in tqdm(testLoader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                _, pred = outputs.topk(5, 1, largest=True, sorted=True)

                label = labels.view(labels.size(0), -1).expand_as(pred)
                correct = pred.eq(label).float()

                #compute top 5
                correct_5 += correct[:, :5].sum()

                #compute top1
                correct_1 += correct[:, :1].sum()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct_1 / total, correct_5 / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--output_dir", type=str, default="../result/seht",
                        help="output directory")


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
    parser.add_argument("--model", type=str, default="resnet18",
                        help="Model name")



    # Dataset setting
    parser.add_argument("--dataset_path", type=str, default='../data',
                        help="where is the datset")
    parser.add_argument("--dataset_name", type=str, default='cifar10',
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
    parser.add_argument("--add_noise", type=int, default=0,
                        help="Whether use noisy trace across each part")
    parser.add_argument("--noise_std", type=int, default=1,
                        help="The standard deviation of noise")
    parser.add_argument("--hess_interval", type=int, default=10,
                        help="The interval to calculate hessian trace")
    parser.add_argument("--lambda_CP", type=float, default=0,
                        help="Coefficieint of Confidence Penalty")
    parser.add_argument("--lambda_LS", type=float, default=0,
                        help="Coefficieint of Label Smoothing")
    parser.add_argument("--alpha_mixup", type=float, default=0,
                        help="alpha for mix up")
    parser.add_argument("--mask_size", type=int, default=0,
                        help="mask size for cutout")
    parser.add_argument("--p_cutout", type=float, default=0,
                        help="p for cutout")
    parser.add_argument("--graph", type=int, default=1,
                        help="draw the graph")
    parser.add_argument("--dropout", type=float, default=0,
                        help="dropout for wide resnet")
    args = parser.parse_args()


    if not os.path.exists(args.output_dir) and args.local_rank in [0, -1]:
        os.makedirs(args.output_dir)

    output_filename = f'{args.output_dir}/random_hessian_{args.dataset_name}_{args.lambda_JR}_{args.Hiter}_{args.hess_interval}_{args.prob}.txt'

    if args.local_rank==-1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl',
                                             timeout=timedelta(minutes=60))

    if args.dataset_name=="cifar10":
        nb_classes=10
    elif args.dataset_name=="cifar100":
        nb_classes=100
    else:
        nb_classes=1000

    trainLoader, testLoader = get_loader(args)
    if "cifar" in args.dataset_name:
        if args.model == "resnet18":
            net = ResNet18(nb_classes).to(device)

        if args.model == "resnet50":
            net = ResNet50(nb_classes).to(device)

        elif args.model == "wide2810":
            net = Wide_ResNet28_10(args, nb_classes).to(device)

    if args.dataset_name=="imagenet":
        if args.model == "resnet18":
            net = resnet18().to(device)

        if args.model == "resnet50":
            net = resnet50().to(device)


    if args.local_rank!=-1:
        # net = DDP(net, message_size=250000000, gradient_predivide_factor=get_world_size(), delay_allreduce=True)
        net = DDP(net)



    for param in net.parameters():
        param.requires_grad =True

    criterion = nn.CrossEntropyLoss()
    cp = ConfidencePenalty()
    ls = LabelSmoothing()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    train_record = []
    test_record = []
    hessian_record = []
    if args.model == "resnet18":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.model == "wide2810":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    start_time = datetime.datetime.now()
    # Train
    correct = 0
    total = 0
    max_test = 0
    max_test_1 = 0
    max_test_5 = 0
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

            # MixUp
            if args.alpha_mixup == 0:
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                loss_super = criterion(outputs, labels)
            else:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, args.alpha_mixup)
                inputs, targets_a, targets_b = map(Variable, (inputs,
                                                          targets_a, targets_b))
                loss_super = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (lam * predicted.eq(targets_a.data).cpu().sum().float()
                    + (1 - lam) * predicted.eq(targets_b.data).cpu().sum().float())
            
            # Calculate trace with hutchinson
            if idx % args.hess_interval == 0:
                trace, hessian_tr = hutchinson(args, net, loss_super, outputs, device)

            # Add Noise
            hloss = torch.normal(args.lambda_JR * (trace / args.Hiter), torch.ones(1).cuda()*args.noise_std)
            # Final Loss
            loss = loss_super + hloss

            # Confidence Penalty / Label Smoothing
            if args.lambda_CP > 0:
                loss = loss + args.lambda_CP * cp(outputs)
            if args.lambda_LS > 0:
                loss = loss + args.lambda_CP * ls(outputs)

            loss.backward()
            optimizer.step()
            with torch.no_grad():
            # print statistics
                running_loss += loss_super.item()
                hessian_loss += hloss.item()

        train_acc = correct / total
        # Validation after training
        if args.dataset_name == "cifar10":
            valid_acc = valid(args, net, testLoader, device)
            if valid_acc >  max_test:
                max_test = valid_acc
            if args.local_rank in [-1, 0]:
                INFO = f'[Epoch {epoch+1}/{args.epochs}] TRAINING Accuracy : ({(100 * train_acc):3f}%) | TEST Accuracy : ({(100 * valid_acc):3f}%) | Hessian Loss : ({(hessian_loss):3f})\n'
                print(INFO)
                with open(output_filename ,'a',encoding='utf-8') as f:
                    f.write(INFO)
        elif args.dataset_name == "cifar100" or args.dataset_name == "imagenet":
            valid_acc, valid_acc_5 = valid(args, net, testLoader, device)
            if valid_acc >  max_test_1:
                max_test_1 = valid_acc
            if valid_acc_5 >  max_test_5:
                max_test_5 = valid_acc_5
            if args.local_rank in [-1, 0]:
                INFO = f'[Epoch {epoch+1}/{args.epochs}] TRAINING Accuracy : ({(100 * train_acc):3f}%) | TEST1 Accuracy : ({(100 * valid_acc):3f}%) | TEST5 Accuracy : ({(100 * valid_acc_5):3f}%) Hessian Loss : ({(hessian_loss):3f})\n'
                print(INFO)
                with open(output_filename ,'a',encoding='utf-8') as f:
                    f.write(INFO)
        scheduler.step()

        train_record.append(train_acc)
        test_record.append(valid_acc.cpu())
        hessian_record.append(hessian_loss)

    if args.local_rank in [-1, 0]:
        if args.dataset_name == "cifar10":
            print('Finished Training, max test accuracy', 100 * max_test)
            with open(output_filename,'a',encoding='utf-8') as f:
                f.write('Best TEST Accuracy inputs: %.3f %% \n' % (100 * max_test))
        elif args.dataset_name == "cifar100" or args.dataset_name=="imagenet":
            print('Finished Training, max test 1 accuracy', 100 * max_test_1)
            print('Finished Training, max test 5 accuracy', 100 * max_test_5)
            with open(output_filename,'a',encoding='utf-8') as f:
                f.write('Best TEST 1 Accuracy inputs: %.3f %% \n' % (100 * max_test_1))
                f.write('Best TEST 5 Accuracy inputs: %.3f %% \n' % (100 * max_test_5))
        end_time = datetime.datetime.now()
        delta = end_time - start_time
        delta_gmtime = time.gmtime(delta.total_seconds())
        duration_str = time.strftime("%H:%M:%S", delta_gmtime)
        print(duration_str)
        if args.graph == 1:
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.plot(train_record,label="train accuracy")
            ax1.plot(test_record,label="test accuracy")
            ax1.set_ylabel('Accuracy')
            ax2 = ax1.twinx() 
            ax2.plot(hessian_record,label='Hessian Trace')
            ax2.set_ylabel('Estimated Hessian Trace')
            ax2.set_xlabel('Epoch')
            plt.legend()
            plt.savefig(f'{args.output_dir}/random_hessian_{args.dataset_name}_{args.lambda_JR}_{args.Hiter}_{args.hess_interval}_{args.prob}.png')

if __name__ == "__main__":
    main() 
