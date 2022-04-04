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



def main():
    lr=1e-2
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
        transforms.RandomCrop(32, padding=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    transform_test = transforms.Compose([
        transforms.RandomCrop(32, padding=2),
        transforms.ToTensor(),
        normalize
    ])
    dataset1 = datasets.FashionMNIST('../data', train=True, download=True,
                                transform=transform_train)
    dataset2 = datasets.FashionMNIST('../data', train=False,
                                transform=transform_test)
    train_dataloader = torch.utils.data.DataLoader(dataset1,batch_size=64, shuffle=True, num_workers=16,pin_memory = True)
    test_dataloader = torch.utils.data.DataLoader(dataset2, batch_size=64, num_workers=16,pin_memory = True)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1)
            self.conv2 = nn.Conv2d(32, 64, 3, 1, padding=1)
            self.conv3 = nn.Conv2d(64, 64, 3, 1, padding=1)
            self.conv4 = nn.Conv2d(64, 128, 3, 1, padding=1)
            self.fc = nn.Linear(512, 10)

        def forward(self, x):
            x = self.conv1(x)
            x = F.max_pool2d(F.relu(x),2)#16*16*32
            x = self.conv2(x)
            x = F.max_pool2d(F.relu(x),2)#8*8*64
            x = self.conv3(x)
            x = F.max_pool2d(F.relu(x), 2)  # 4*4*64
            x = self.conv4(x)
            x = F.max_pool2d(F.relu(x), 2)  # 2*2*128
            x = x.view(-1,512)
            output = self.fc(x)
            return output

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
        for batch_idx, (inputs, targets) in enumerate(train_loader):

            # torch.cuda.synchronize()
            # since = time.time()
            # if device == 'cuda':
            #     inputs, targets = inputs.cuda(), targets.cuda(non_blocking=False)  # async=True)
            # inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
            # inputs = Variable(inputs, requires_grad=True).to(device)
            inputs = inputs.cuda(non_blocking=True)
            # targets = Variable(targets).to(device)
            targets = targets.cuda(non_blocking=True)

            # compute output
            outputs = model(inputs)

            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_idx % 100 == 0:
                # print('({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                print('train Epoch: [{0}][{1}/{2}]\t'
                # 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                # 'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss:.4f} ({loss:.4f})\t'
                      'Prec@1 {top1:.3f} ({top1:.3f})\t'
                      'Prec@5 {top5:.3f} ({top5:.3f})\t'.format(
                    epoch, batch_idx, len(train_loader),  # batch_time=batch_time,
                    # data_time=data_time,
                    loss=losses.avg, top1=top1.avg, top5=top5.avg))
            # print('Time {batch_time.val:.3f}\t''Data {data_time.val:.3f}\t'.format(batch_time=batch_time, data_time=data_time))
            # print("loss",loss.data)
            # train_loss=loss.data
        # return batch_time.avg,data_time.avg,time_elapsed
        # return time_elapsed


    def test(test_loader, model, criterion, epoch):
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        model.eval()

        with torch.no_grad():
            # end = time.time()
            for batch_idx, (inputs, targets) in enumerate(test_loader):
                # if device == 'cuda':
                #     inputs, targets = inputs.cuda(), targets.cuda()
                # inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)
                inputs = inputs.cuda(non_blocking=True)
                # inputs = inputs.to(device)
                targets = targets.cuda(non_blocking=True)
                # compute output
                outputs = model(inputs)

                loss = criterion(outputs, targets)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
                losses.update(loss.item(), inputs.size(0))
                top1.update(prec1.item(), inputs.size(0))
                top5.update(prec5.item(), inputs.size(0))

                # measure elapsed time
                # batch_time.update(time.time() - end)
                # end = time.time()

            # TODO: this should also be done with the ProgressMeter
            print('test Epoch: [{0}]\t'
                  ' * Acc@1 {top1:.3f} Acc@5 {top5:.3f}'
                  .format(epoch, top1=top1.avg, top5=top5.avg))



    model = Net()
    load_model = 'pretrained_model.pth'
    load = torch.load(load_model)
    load_state = {k: v for k, v in load.items() if (k in model.state_dict() or k in model.state_dict())}
    model_state = model.state_dict()
    model_state.update(load_state)
    model.load_state_dict(model_state)
    criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
    model.to(device)
    criterion.to(device)
    for name, param in model.named_parameters():
        if (name !='fc.weight' and name !='fc.bias' ):
            print(name)
            param.requires_grad = False
    #optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    optimizer = optim.Adam(model.fc.parameters(), lr=lr,betas=(0.5, 0.999))#To begin with the task, use Adam with learning rate 2e-4, b1 0.5, and b2 = 0.999

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10,
                                          gamma=0.1)



    for epoch in range(0, 100):
        start_time = timeit.default_timer()
        test(test_dataloader, model, criterion, epoch)
        stop_time = timeit.default_timer()
        print("Execution time: " + str(stop_time - start_time) + "\n")

        # train for one epoch
        start_time = timeit.default_timer()
        train(train_dataloader, model, criterion, optimizer, epoch)
        stop_time = timeit.default_timer()
        print("Execution time: " + str(stop_time - start_time) + "\n")





if __name__ == '__main__':
    main()
