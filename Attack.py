from __future__ import print_function
import argparse
import ast
import os
import shutil
import sys
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torchvision import transforms
import models.bottom_model_plus as models
from my_utils import AverageMeter, accuracy, mkdir_p, precision_recall, train_val_split
import datasets.get_dataset as get_dataset
import dill
import copy
from vfl_framework import VflFramework
from vfl_framework_for_idc import IdcVflFramework
from models import model_sets
from models.model_sets import BottomModel
from my_utils.utils import weights_init, image_format_2_rgb
from PIL import Image
torch.cuda.init()
parser = argparse.ArgumentParser(description='Model Completion')
# dataset paras
parser.add_argument('--dataset-name', default="Criteo", type=str,
                    choices=['CIFAR10', 'CIFAR100', 'TinyImageNet', 'CINIC10L', 'BC_IDC', 'Criteo', 'BCW'])
parser.add_argument('--dataset-path', default='D:/Datasets/Criteo/criteo.csv', type=str)

# attacker's knowledge paras
parser.add_argument('--n-labeled', type=int, default=100,
                    help='Number of labeled data')  # cifar-40, liver-10, TinyImageNet-600
# inference head paras
parser.add_argument('--num-layer', type=int, default=1,
                    help='number of layers of the inference head')
parser.add_argument('--use-bn', type=ast.literal_eval, default=False,
                    help='Inference head use batchnorm or not')
parser.add_argument('--activation_func_type', type=str, default='None',
                    help='Activation function type of the inference head',
                    choices=['ReLU', 'Sigmoid', 'None'])
# vfl paras
parser.add_argument('--party-num', help='party-num',
                    type=int, default=2)
parser.add_argument('--half', help='number of the adversary\'s features. For image datasets except IDC, this para '
                                   'represents number of pixels. For IDC, this para represents number of images( in'
                                   'this case it must be lower than party-num. For numeric datasets, it represents'
                                   'number of numeric features.'
                                   'You can change this para (lower that party_num) to evaluate the sensitivity of our '
                                   'attack.',
                    type=int, default=4096)  # choices=[16, 14, 32, 1->party_num]. CIFAR10-16, Liver-14, TinyImageNet-32
# checkpoints paras (used for trained bottom model in our attack)
parser.add_argument('--resume-dir',
                    default='./FLSG_saved_experiment_results-cinic-10_half16/saved_models/',
                    type=str, metavar='PATH',
                    help='path to latest checkpoint', )
parser.add_argument('--save-dir', dest='save_dir',
                        help='The directory used to save the trained models and csv files',
                        default='./FLSG_saved_experiment_results-cinic-10', type=str)
parser.add_argument('--resume-name',
                    default='Criteo_saved_framework_lr=0.05_normal_half=4096.pth',
                    type=str, metavar='NAME',
                    help='file name of the latest checkpoint', )
parser.add_argument('--out', default=None,  # 'result'
                    help='Directory to output the best checkpoint')
# evaluation paras
parser.add_argument('--k', help='top k accuracy',
                    type=int, default=2)
# training paras
parser.add_argument('--batch-size', default=16, type=int, metavar='N',
                    help='train batchsize')  # CIFAR10/ImageNet-32
parser.add_argument('--epochs', default=10, type=int, metavar='N',
                    help='number of total epochs to run')  # cifar-5, BC_IDC-1, liver-5
parser.add_argument('--lr', '--learning-rate', default=2e-3, type=float,
                    metavar='LR', help='initial learning rate')  # CIFAR10 2e-3
parser.add_argument('--gpu', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--manualSeed', type=int, default=0, help='manual seed')
parser.add_argument('--val-iteration', type=int, default=1024,#1024
                    help='Number of labeled data')
parser.add_argument('--alpha', default=0.75, type=float)
parser.add_argument('--lambda-u', default=50, type=float)
parser.add_argument('--T', default=0.8, type=float)
parser.add_argument('--ema-decay', default=0.999, type=float)
# print paras
parser.add_argument('--print-to-txt', default=0, type=int, choices=[0, 1], help='save all outputs to txt or not')
parser.add_argument('-d', '--dataset', default='CINIC10L', type=str,
                        help='name of dataset',
                        choices=['CIFAR10', 'CIFAR100', 'TinyImageNet', 'CINIC10L', 'Yahoo', 'Criteo', 'BCW', 'BC_IDC'])
parser.add_argument('--usegrad', default=False, type=bool)
parser.add_argument('--trainAtoB', default=False, type=bool)
parser.add_argument('--grad-size', default=0, type=int)
parser.add_argument('--grad-sub-method', default='pca', type=str)

args = parser.parse_args()
args.resume = args.save_dir + f"_half{args.half}/saved_models/" + f'{args.dataset_name}_saved_models/' + args.resume_name
args.out = args.save_dir + f"_half{args.half}/saved_models/" + f'{args.dataset_name}_saved_models/'
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_cuda = torch.cuda.is_available()


# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
np.random.seed(args.manualSeed)

best_acc = 0  # best test accuracy


class EnhancedSet(data.Dataset):
    def __init__(self, X, Y, transform=None):
        self.X, self.Y = X, Y
        if args.dataset == 'CINIC10L' and transform != None:
            self.transform = self.get_transforms()
        else:
            self.transform = transform

    def __getitem__(self, index):
        if args.dataset == 'CINIC10L' and self.transform != None:
            file_path, label = self.X[index]
            img = self.read_image(file_path)
            return img, self.Y[index]
        else:
            img, target = self.X[index], self.Y[index]
            if self.transform is not None:
                img = self.transform(img)
            return img, target

    def __len__(self):
        return len(self.X)
    
    def read_image(self, path):
        img = Image.open(path)
        return self.transform(img) if self.transform else img
    
    def get_normalize_transform(self):
        normalize_cinic = transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404],
                                               std=[0.24205776, 0.23828046, 0.25874835])
        return normalize_cinic

    def get_transforms(self):
        normalize = self.get_normalize_transform()
        transforms_ = transforms.Compose([
            transforms.Lambda(image_format_2_rgb),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            normalize
        ])
        return transforms_



def enhanced_data(dataset_setup, train_complete_dataset, train_labeled_set, train_unlabeled_set, train_labeled_idxs, train_unlabeled_idxs):
    df = pd.read_csv(args.save_dir + f"_half{args.half}/csv_files/{args.dataset}_csv_files/{args.dataset}_imdgrad_{args.grad_sub_method}_lr=0.1_normal_half={args.half}_59.csv")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.resume, map_location=device, pickle_module=dill)


    if args.dataset_name == "BC_IDC" or args.dataset_name == "covid":
        bottom_model = copy.deepcopy(checkpoint.bottom_models[0])
    else:
        bottom_model = copy.deepcopy(checkpoint.malicious_bottom_model_a)

    enhanced = None
    labels = None
    data_loader = data.DataLoader(train_labeled_set, batch_size=256, shuffle=False,
                                              num_workers=0,
                                              drop_last=False)
    for batch_idx, (datas, target) in enumerate(data_loader):
        inputs = dataset_setup.clip_one_party_data(datas, args.half)
        inputs = inputs.type(torch.float)
        inputs = inputs.cuda()
        y = bottom_model(inputs)
        if enhanced == None:
            enhanced = y.tolist()
        else:
            enhanced.extend(y.tolist())
        if labels == None:
            labels = target.tolist()
        else:
            labels.extend(target.tolist())
    for i,d in enumerate(train_labeled_idxs):
        for j in range(args.grad_size):
            enhanced[i].append(df[str(j)][d])
        
    train_labeled_enhanced_data = torch.Tensor(enhanced)
    train_labeled_enhanced_target = torch.Tensor(labels)

    enhanced = None
    labels = None
    data_loader = data.DataLoader(train_unlabeled_set, batch_size=256, shuffle=False,
                                              num_workers=0,
                                              drop_last=False)
    for batch_idx, (datas, target) in enumerate(data_loader):
        inputs = dataset_setup.clip_one_party_data(datas, args.half)
        inputs = inputs.type(torch.float)
        inputs = inputs.cuda()
        y = bottom_model(inputs)
        if enhanced == None:
            enhanced = y.tolist()
        else:
            enhanced.extend(y.tolist())
        if labels == None:
            labels = target.tolist()
        else:
            labels.extend(target.tolist())
    for i,d in enumerate(train_unlabeled_idxs):
        for j in range(args.grad_size):
            enhanced[i].append(df[str(j)][d])  
    train_unlabeled_enhanced_data = torch.Tensor(enhanced)
    train_unlabeled_enhanced_target = torch.Tensor(labels)

    enhanced = None
    labels = None
    data_loader = data.DataLoader(train_complete_dataset, batch_size=256, shuffle=False,
                                              num_workers=0,
                                              drop_last=False)
    for batch_idx, (datas, target) in enumerate(data_loader):
        inputs = dataset_setup.clip_one_party_data(datas, args.half)
        inputs = inputs.type(torch.float)
        inputs = inputs.cuda()
        y = bottom_model(inputs)
        if enhanced == None:
            enhanced = y.tolist()
        else:
            enhanced.extend(y.tolist())
        if labels == None:
            labels = target.tolist()
        else:
            labels.extend(target.tolist())
    for i in range(len(train_complete_dataset)):
        for j in range(args.grad_size):
            enhanced[i].append(df[str(j)][i]) 
    train_complete_enhanced_data = torch.Tensor(enhanced)
    train_complete_enhanced_target = torch.Tensor(labels)

    labeled_trainset_enhanced = EnhancedSet(train_labeled_enhanced_data,train_labeled_enhanced_target)
    unlabeled_trainset_enhanced = EnhancedSet(train_unlabeled_enhanced_data,train_unlabeled_enhanced_target)
    complete_trainset_enhanced = EnhancedSet(train_complete_enhanced_data,train_complete_enhanced_target)

    return labeled_trainset_enhanced, unlabeled_trainset_enhanced, complete_trainset_enhanced


def random_train_loader(train_dataset):
    indices = list(range(len(train_dataset)))
    train_dataset_ = torch.utils.data.Subset(train_dataset,indices)
    random.shuffle(indices)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset_,
        batch_size=256, 
        sampler=torch.utils.data.sampler.SequentialSampler(indices),
    )

    return train_loader, indices


def main():
    if args.batch_size > args.n_labeled:
        raise Exception("args.batch_size must be smaller than args.n_labeled")

    global best_acc
    if args.out and not os.path.isdir(args.out):
        mkdir_p(args.out)

    print(args)

    # datasets settings
    print('==> Preparing {}'.format(args.dataset_name))
    dataset_setup = get_dataset.get_dataset_setup_by_name(args.dataset_name)
    size_bottom_out = dataset_setup.size_bottom_out
    num_classes = dataset_setup.num_classes
    clip_function = dataset_setup.clip_one_party_data
    zip_ = dataset_setup.set_datasets_for_ssl(file_path=args.dataset_path,
                                            n_labeled=args.n_labeled, party_num=args.party_num)
    train_labeled_set, train_unlabeled_set, test_set, train_complete_dataset = zip_
    train_labeled_idxs = dataset_setup.train_labeled_idxs
    train_unlabeled_idxs = dataset_setup.train_unlabeled_idxs


    if args.dataset_name == 'Criteo':
        labeled_trainloader, unlabeled_trainloader, test_loader, train_complete_trainloader = zip_
    else:

        labeled_trainset, unlabeled_trainset, train_complete_trainset = enhanced_data(dataset_setup, train_complete_dataset, train_labeled_set, train_unlabeled_set, train_labeled_idxs, train_unlabeled_idxs)
        labeled_trainloader = data.DataLoader(labeled_trainset, batch_size=args.batch_size, shuffle=True,
                                              num_workers=0,
                                              drop_last=True)
        unlabeled_trainloader = data.DataLoader(unlabeled_trainset, batch_size=args.batch_size, shuffle=True,
                                                num_workers=0, drop_last=True)
        dataset_bs = args.batch_size * 10
        train_complete_trainloader = data.DataLoader(train_complete_trainset, batch_size=dataset_bs, shuffle=False,
                                                     num_workers=0, drop_last=False)

    # Model
    print("==> creating bottom model plus")

    def create_model(ema=False, size_bottom_out=10, num_classes=10):
        model = models.BottomModelEnhance(size_bottom_out, num_classes,
                                       num_layer=args.num_layer,
                                       activation_func_type=args.activation_func_type,
                                       use_bn=args.use_bn)
        model = model.cuda()

        if ema:
            for param in model.parameters():
                param.detach_()

        return model

    model = create_model(ema=False, size_bottom_out=size_bottom_out+args.grad_size, num_classes=num_classes)
    ema_model = create_model(ema=True, size_bottom_out=size_bottom_out+args.grad_size, num_classes=num_classes)

    cudnn.benchmark = True

    train_criterion = SemiLoss()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    ema_optimizer = WeightEMA(model, ema_model, alpha=args.ema_decay)

    # Resume
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        if args.out:
            args.out = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume, pickle_module=dill)
        # model.fc.apply(weights_init_ones)
        # ema_model.fc.apply(weights_init_ones)
        if args.dataset_name == "BC_IDC" or args.dataset_name == "covid":
            model.bottom_model = copy.deepcopy(checkpoint.bottom_models[0])
            ema_model.bottom_model = copy.deepcopy(checkpoint.bottom_models[0])
        else:
            # print("checkpoint:", checkpoint.malicious_bottom_model_a)
            model.bottom_model = copy.deepcopy(checkpoint.malicious_bottom_model_a)
            ema_model.bottom_model = copy.deepcopy(checkpoint.malicious_bottom_model_a)

    if args.dataset_name == 'Criteo':
        for param in model.bottom_model.parameters():
            param.requires_grad = False
    print("---Label inference on complete training dataset:")

    for epoch in range(args.epochs):
        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_loss_x, train_loss_u = train(labeled_trainloader, unlabeled_trainloader, model, optimizer,
                                                       ema_optimizer, train_criterion, epoch, use_cuda,
                                                       clip_function, num_classes)
        print("---Label inference on complete training dataset:")
        _, train_acc = validate(train_complete_trainloader, ema_model, criterion, epoch, use_cuda, mode='Train Stats',
                                num_classes=num_classes, clip_function=clip_function)

        # save model
        is_best = train_acc > best_acc
        best_acc = max(train_acc, best_acc)
        if args.out:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'ema_state_dict': ema_model.state_dict(),
                'train_acc': train_acc,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best)


    print('Best top 1 accuracy:')
    print(best_acc)

    targets_v = []
    for batch_idx, (inputs, targets) in enumerate(train_complete_trainloader):
        inputs = inputs.cuda()
        inputs = inputs.type(torch.float)
        targets = ema_model(inputs)
        targets_v.extend(torch.argmax(targets,dim=1).view(-1).tolist())            

    if args.dataset_name == "BC_IDC" or args.dataset_name == "covid":
        bottom_model_a = copy.deepcopy(checkpoint.bottom_models[0])
        bottom_model_b = copy.deepcopy(checkpoint.bottom_models[1])
        top_model = copy.deepcopy(checkpoint.top_model)
    else:
        bottom_model_a = copy.deepcopy(checkpoint.malicious_bottom_model_a)
        bottom_model_b = copy.deepcopy(checkpoint.benign_bottom_model_b)
        top_model = copy.deepcopy(checkpoint.top_model)
    
    t = 3
    s = 6
    targets_v = np.array(targets_v)
    targets = np.array(train_complete_dataset.targets)

    target_virtual_tensor = torch.Tensor(targets_v)
    virtual_traindataset = EnhancedSet(train_complete_dataset.data, target_virtual_tensor, transform=transforms.ToTensor())
    all_grad=[]
    all_outputA=[]
    if args.usegrad:
        df_grad = pd.read_csv(args.save_dir + f"csv_files/{args.dataset}_csv_files/{args.dataset}_imdgrad_all.csv")
        for i in range(100):
            all_grad.append(np.array(df_grad.iloc[:,100*i+1:100*i+101]))

        df_outputA = pd.read_csv(args.save_dir + f"csv_files/{args.dataset}_csv_files/{args.dataset}_outputA_all.csv")
        for i in range(100):
            all_outputA.append(np.array(df_outputA.iloc[:,100*i+1:100*i+101]))
    
    df_grad = None
    df_outputA = None

    suc = 0
    fail = 0
    others = 0
    posion_input = None
    virtual_top_model = model_sets.TopModel(dataset_name=args.dataset).get_model()
    virtual_top_model = virtual_top_model.cuda()
    optimizer_top = optim.SGD(virtual_top_model.parameters(), lr=0.01)
    if args.trainAtoB:
        in_A_to_Out_B = model_sets.BottomModel(dataset_name=args.dataset).get_model(
                half=args.half,
                is_adversary=True
            )
        in_A_to_Out_B = in_A_to_Out_B.cuda()
        optimizer_outb = optim.SGD(in_A_to_Out_B.parameters(), lr=0.1)

    input_a = None

    virtual_top_model.train()
    bottom_model_a.eval()
    bottom_model_b.eval()
    criterion = torch.nn.CrossEntropyLoss()
    criterion_grad = torch.nn.CrossEntropyLoss()
    ep = 50

    print('start simulate')
    for i in range(ep):

        virtual_trainloader,indices = random_train_loader(virtual_traindataset)
        start_idx=0
        end_idx = 0
        lamm = 0.1
        for batch_idx, (inputs, targets) in enumerate(virtual_trainloader):
            input_a = inputs[:, :, :, 0:args.half]
            input_a = input_a.float().cuda()
            out_a = bottom_model_a(input_a)
            if args.trainAtoB:
                outb_batch = in_A_to_Out_B(input_a)
            else:
                outb_batch = torch.randn_like(out_a).cuda()
            out = virtual_top_model(out_a, outb_batch) 
            targets = targets.long().cuda()
            loss = criterion(out, targets)
            
            if args.usegrad:
                end_idx += len(targets)
                g_target = torch.tensor(all_grad[i*2][indices[start_idx:end_idx]]).cuda().float()
                outA_forgrad = torch.tensor(all_outputA[i*2][indices[start_idx:end_idx]]).cuda().float()
                outA_forgrad.requires_grad_(True)
                out_forgrad = virtual_top_model(outA_forgrad, outb_batch) 
                loss_forgrad= criterion(out_forgrad, targets)
                start_idx = end_idx
                g = torch.autograd.grad(outputs=loss_forgrad, inputs=outA_forgrad, create_graph=True)[0].float()
                loss_grad = criterion_grad(g,g_target).float()
                #print(loss_grad.item())
                
                loss+=lamm * loss_grad / ((loss_grad/loss).detach())
            #loss += lamm * loss_grad
            if args.trainAtoB:
                optimizer_outb.zero_grad()
            optimizer_top.zero_grad()
            
            loss.backward()

            if args.trainAtoB:
                optimizer_outb.step()
            optimizer_top.step()
        
    virtual_top_model.eval()
    if args.trainAtoB:
        in_A_to_Out_B.eval()

    test_loader,indices = random_train_loader(test_set)
    correct = 0
    total = 0
    kl = 0 
    count = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            input_a = inputs[:, :, :, 0:args.half].float().cuda()
            out_a = bottom_model_a(input_a)
            input_b = inputs[:, :, :, args.half:32].float().cuda()
            out_b = bottom_model_b(input_b)
            
            out_v = virtual_top_model(out_a, out_b)
            out_t = top_model(out_a, out_b)
            kl += torch.nn.functional.kl_div(out_v.softmax(dim=-1).log(), out_t.softmax(dim=-1), reduction='mean')
            count +=1
                                       
        print(kl/count)

    bottom_model_a.train()
    optimizer = optim.SGD(bottom_model_a.parameters(), lr=0.01)
    
    print("poison")
    for i in range(30):
        if i == 20:
            for param_group in optimizer.param_groups:
                param_group["lr"] = 0.001
        virtual_trainloader,indices = random_train_loader(virtual_traindataset)
        start_idx=0
        end_idx = 0
        for batch_idx, (inputs, targets) in enumerate(virtual_trainloader):
            input_a = inputs[:, :, :, 0:args.half]
            input_a = input_a.float()
            input_a=input_a.cuda()
            out_a = bottom_model_a(input_a)
            if args.trainAtoB:
                outb_batch = in_A_to_Out_B(input_a)
            else:
                outb_batch = torch.randn_like(out_a).cuda()
 
            out = virtual_top_model(out_a, outb_batch) 
            targets = targets.long()
            for j in range(len(targets)):
                if targets[j] == s or True:
                    targets[j] = t
            targets = targets.cuda()
            loss = criterion(out, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    bottom_model_a.eval()
        
    print('test')
    for img_idx, (img, truelabel) in enumerate(test_set):
        if truelabel == s or True:
            input_a = img[:, :, 0:args.half]
        
            input_a=torch.unsqueeze(input_a, 0)
            input_a=input_a.cuda()

            input_b = img[:, :, args.half:32]
            input_b=torch.unsqueeze(input_b, 0)
            input_b=input_b.cuda()

            out_a = bottom_model_a(input_a)
            out_b = bottom_model_b(input_b)
            out = top_model(out_a, out_b)
            out=torch.argmax(out,dim=1)
            if out[0] == t:
                suc+=1
            elif out[0] == truelabel:
                fail+=1
            else:
                others+=1
    print(suc,fail,others)
    with open('data.txt', 'a') as f:
        f.write(str(suc)+','+str(fail)+','+str(others))
        f.write('\n')
    

def train(labeled_trainloader, unlabeled_trainloader, model, optimizer, ema_optimizer, criterion, epoch, use_cuda,
          clip_function, num_classes):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_u = AverageMeter()
    ws = AverageMeter()
    end = time.time()

    labeled_train_iter = iter(labeled_trainloader)
    unlabeled_train_iter = iter(unlabeled_trainloader)

    model.train()

    for batch_idx in range(args.val_iteration):
        if args.dataset_name == 'Criteo':
            if batch_idx == 1000 - args.n_labeled:
                break
            inputs_x, targets_x = labeled_trainloader[batch_idx % len(labeled_trainloader)]
            inputs_u, _ = unlabeled_trainloader[batch_idx % len(unlabeled_trainloader)]
        else:
            try:
                inputs_x, targets_x = next(labeled_train_iter)
                # inputs_x, targets_x = labeled_trainloader.dataset[batch_idx]
            except StopIteration:
                labeled_train_iter = iter(labeled_trainloader)
                inputs_x, targets_x = next(labeled_train_iter)
            try:
                inputs_u, _ = next(unlabeled_train_iter)
            except StopIteration:
                unlabeled_train_iter = iter(unlabeled_trainloader)
                inputs_u, _ = next(unlabeled_train_iter)

        # measure data loading time
        data_time.update(time.time() - end)

        # in vertical federated learning scenario, attacker(party A) only has part of features, i.e. half of the img.
        inputs_x = inputs_x.type(torch.float)
        inputs_u = inputs_u.type(torch.float)

        batch_size = inputs_x.size(0)

        # Transform label to one-hot
        targets_x = targets_x.view(-1, 1).type(torch.long)
        targets_x = torch.zeros(batch_size, num_classes).scatter_(1, targets_x, 1)

        if use_cuda:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda(non_blocking=True)
            inputs_u = inputs_u.cuda()

        with torch.no_grad():
            targets_x.view(-1, 1).type(torch.long)  # compute guessed labels of unlabel samples
            outputs_u = model(inputs_u)
            p = torch.softmax(outputs_u, dim=1)
            pt = p ** (1 / args.T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        # mixup
        all_inputs = torch.cat([inputs_x, inputs_u], dim=0)
        all_targets = torch.cat([targets_x, targets_u], dim=0)

        l = np.random.beta(args.alpha, args.alpha)

        l = max(l, 1 - l)

        idx = torch.randperm(all_inputs.size(0))

        input_a, input_b = all_inputs, all_inputs[idx]
        target_a, target_b = all_targets, all_targets[idx]

        mixed_input = l * input_a + (1 - l) * input_b
        mixed_target = l * target_a + (1 - l) * target_b

        # interleave labeled and unlabeled samples between batches to get correct batch norm calculation
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        logits = [model(mixed_input[0])]
        for input in mixed_input[1:]:
            logits.append(model(input))

        # put interleaved samples back
        logits = interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)

        Lx, Lu, w = criterion(logits_x, mixed_target[:batch_size], logits_u, mixed_target[batch_size:],
                              epoch + batch_idx / args.val_iteration)

        loss = Lx + w * Lu

        # record loss
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_u.update(Lu.item(), inputs_x.size(0))
        ws.update(w, inputs_x.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # print('one batch training done')
        if batch_idx % 250 == 0:
            print("batch_idx:", batch_idx, " loss:", losses.avg)
    return losses.avg, losses_x.avg, losses_u.avg


def validate(valloader, model, criterion, epoch, use_cuda, mode, num_classes, clip_function):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    topk = AverageMeter()
    precision = AverageMeter()
    recall = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            # in vertical federated learning scenario, attacker(party A) only has part of features, i.e. half of the img

            # measure data loading time
            data_time.update(time.time() - end)

            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda(non_blocking=True)
            # compute output
            inputs = inputs.type(torch.float)
            outputs = model(inputs)
            targets = targets.type(torch.long)
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, preck = accuracy(outputs, targets, topk=(1, args.k))
            if num_classes == 2:
                prec, rec = precision_recall(outputs, targets)
                precision.update(prec, inputs.size(0))
                recall.update(rec, inputs.size(0))
                # print("batch_id", batch_idx, end='')
                # print(" precision", precision.avg, end='')
                # print(", recall", recall.avg, end='')
                # print("  F1", 2 * (precision.avg * recall.avg) / (precision.avg + recall.avg))

            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            topk.update(preck.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            # print('one batch done')
    print("Dataset Overall Statistics:")
    if num_classes == 2:
        print("  precision", precision.avg, end='')
        print("  recall", recall.avg, end='')
        if (precision.avg + recall.avg) != 0:
            print("  F1", 2 * (precision.avg * recall.avg) / (precision.avg + recall.avg), end='')
        else:
            print(f"F1:0")
    print("top 1 accuracy:{}, top {} accuracy:{}".format(top1.avg, args.k, topk.avg))
    return losses.avg, top1.avg


def save_checkpoint(state, is_best, checkpoint=args.out, filename=f'{args.dataset_name}_mc_checkpoint.pth'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, f'{args.dataset_name}_mc_best.pth'))


def linear_rampup(current, rampup_length=args.epochs):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch):
        probs_u = torch.softmax(outputs_u, dim=1)

        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, args.lambda_u * linear_rampup(epoch)


class WeightEMA(object):
    def __init__(self, model, ema_model, alpha=0.999):
        self.model = model
        self.ema_model = ema_model
        self.alpha = alpha
        self.params = list(model.state_dict().values())
        self.ema_params = list(ema_model.state_dict().values())
        self.wd = 0.02 * args.lr

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            ema_param = ema_param.type(torch.float)
            ema_param.mul_(self.alpha)
            ema_param.add_(param * one_minus_alpha)
            # customized weight decay
            param = param.type(torch.float)
            param.mul_(1 - self.wd)


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


if __name__ == '__main__':
    if args.print_to_txt == 0:
        main()
    else:
        inference_head_setting_str = f'_layer={args.num_layer}' \
                                     f'_func={args.activation_func_type}' \
                                     f'_bn={args.use_bn}' \
                                     f'_nlabeled={args.n_labeled}'
        txt_name = 'posion' + args.resume_name + inference_head_setting_str + '.txt'
        savedStdout = sys.stdout
        with open(args.resume_dir + f'{args.dataset_name}_saved_models/' + txt_name, 'w+') as file:
            sys.stdout = file
            main()
            sys.stdout = savedStdout
        print('Results saved to txt!')
