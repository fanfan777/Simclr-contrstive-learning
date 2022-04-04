import timeit
from datetime import datetime
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

def main():
    lr=1e-4
    temperature=0.1#0.5
    batch_size=128
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)


    normalize = transforms.Normalize((0.1307,), (0.3081,))
    # transform=transforms.Compose([
    #     transforms.ToTensor(),
    #     #transforms.Normalize((0.1307,), (0.3081,))
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    #     #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    #     #transforms.Normalize((0.5, 0.5, 0.5), (0.225, 0.225, 0.225))
    #     ])
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(32,scale=(0.8,1.2),ratio=(0.8,1.2)),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4)], p=0.8),
        transforms.ToTensor(),
        normalize
    ])
    path = "./result/"

    class dataset(Dataset):
        def __init__(self, Dataset,transform=transform_train):
            self.transform=transform
            self.data=Dataset.data
            #print("data",self.data)
            #print(Dataset.keys())
            #self.datalist=self.data[:,0,0].tolist()
            #print(len(self.datalist))

        def __len__(self):
            return len(self.data[:,0,0].tolist())

        def __getitem__(self, index):
            img = np.array(self.data[index])
            #print(img)
            img = Image.fromarray(img)
            #print(img)

            if self.transform is not None:
                pos_1 = self.transform(img)
                pos_2 = self.transform(img)

            return pos_1, pos_2

    dataset1 = datasets.FashionMNIST('../data', train=True, download=True)
    train_data = dataset(Dataset=dataset1, transform=transform_train)
    train_dataloader = torch.utils.data.DataLoader(train_data,batch_size=batch_size, shuffle=True, num_workers=16,pin_memory = True,drop_last=True)


    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
            self.conv3 = nn.Conv2d(64, 64, 3, 1, padding=1)
            self.conv4 = nn.Conv2d(64, 128, 3, 1, padding=1)
            self.fc1 = nn.Linear(512, 512)
            self.fc2 = nn.Linear(512, 512)

        def f(self, x):
            x = self.conv1(x)
            x = F.max_pool2d(F.relu(x), 2)  # 16*16*32
            x = self.conv2(x)
            x = F.max_pool2d(F.relu(x), 2)  # 8*8*64
            x = self.conv3(x)
            x = F.max_pool2d(F.relu(x), 2)  # 4*4*64
            x = self.conv4(x)
            features = F.max_pool2d(F.relu(x), 2)  # 2*2*128
            return features

        def g(self, x):
            out=self.fc2(F.relu(self.fc1(x)))
            return out

        def forward(self, x):
            features=self.f(x).view(-1,512)
            out=self.g(features)
            return F.normalize(features, dim=-1), F.normalize(out, dim=-1)

    class AverageMeter(object):
        """Computes and stores the average and current value
           Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
        """

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
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


    def adjust_learning_rate(optimizer, epoch, init_lr):
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = init_lr * (0.1 ** (epoch // 30))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    def train(train_loader, model, criterion, optimizer, epoch):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        model.train()
        # end=time.time()
        # 遍历训练集, 梯度下降训练的框架
        for batch_idx, (pos_1, pos_2,) in enumerate(train_loader):
            #if(batch_idx==0 and epoch==0):
            #print("pos1",pos_1.tolist(),"pos2",pos_2.tolist())
            # torch.cuda.synchronize()
            # since = time.time()
            # if device == 'cuda':
            #     inputs, targets = inputs.cuda(), targets.cuda(non_blocking=False)  # async=True)
            # inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            # inputs = Variable(inputs, requires_grad=True).to(device)
            pos_1 = pos_1.cuda(non_blocking=True)
            # targets = Variable(targets).to(device)
            pos_2 = pos_2.cuda(non_blocking=True)
            feature_1, out_1 = model(pos_1)
            feature_2, out_2 = model(pos_2)
            Imagerevert_1 = pos_1 * 0.3081 + 0.1307
            Imagerevert_2 = pos_2 * 0.3081 + 0.1307
            trans_pos1 = transforms.ToPILImage()(Imagerevert_1[0])
            #trans_pos1.save(path+"1"+str(batch_idx)+".png")
            trans_pos2 = transforms.ToPILImage()(Imagerevert_2[0])
            #trans_pos2.save(path+"2"+str(batch_idx)+".png")
            
            #print("pos1", out_1.tolist(), "pos2", out_2.tolist())
            # [2*B, D]
            out = torch.cat([out_1, out_2], dim=0)
            # [2*B, 2*B]
            sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
            mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
            # [2*B, 2*B-1]
            sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)

            # compute loss
            pos_sim = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
            # [2*B]
            pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
            loss = (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()


            # measure accuracy and record loss
            #prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), pos_1.size(0))
            #top1.update(prec1.item(), inputs.size(0))
            #top5.update(prec5.item(), inputs.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                # print('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                print('train Epoch: [{0}][{1}/{2}]\t'
                # 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                # 'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss:.4f} ({loss:.4f})\t'.format(
                    epoch, batch_idx, len(train_loader),  # batch_time=batch_time,
                    # data_time=data_time,
                    loss=losses.avg))
            # print('Time {batch_time.val:.3f}\t''Data {data_time.val:.3f}\t'.format(batch_time=batch_time, data_time=data_time))
            # print("loss",loss.data)
            # train_loss=loss.data
        # return batch_time.avg,data_time.avg,time_elapsed
        # return time_elapsed


    model = Net().to(device)
    criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
    model.to(device)
    criterion.to(device)
    #optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    optimizer = optim.Adam(model.parameters(), lr=lr,betas=(0.5, 0.999))#To begin with the task, use Adam with learning rate 2e-4, b1 0.5, and b2 = 0.999

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                          gamma=0.1)



    for epoch in range(0, 100):
        start_time = timeit.default_timer()
        train(train_dataloader, model, criterion, optimizer,epoch)
        stop_time = timeit.default_timer()
        print("Execution time: " + str(stop_time - start_time) + "\n")

    torch.save(model.state_dict(), 'pretrained_model.pth')





if __name__ == '__main__':
    main()
