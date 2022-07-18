# 训练DeepSORT的表观特征模型
import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from utilsss import MyDataset, validate, show_confMat
from torch.utils.data import DataLoader

from model import Net

parser = argparse.ArgumentParser(description="Train on market1501")
parser.add_argument("--data-dir",default='data',type=str)
parser.add_argument("--no-cuda",action="store_true")
parser.add_argument("--gpu-id",default=0,type=int)
parser.add_argument("--lr",default=0.1, type=float)
parser.add_argument("--interval",'-i',default=20,type=int)
parser.add_argument('--resume', '-r',action='store_true')
args = parser.parse_args()

#batch_size
train_bs = 32
valid_bs = 32

# device
# 决定使用CPU或GPU进行训练
device = "cuda:{}".format(args.gpu_id) if torch.cuda.is_available() and not args.no_cuda else "cpu"
if torch.cuda.is_available() and not args.no_cuda:
    cudnn.benchmark = True

# data loading
print('==> Preparing data..')

# 数据增强-对图片进行随机翻转等操作
transform = [transforms.RandomHorizontalFlip(p=0.5),transforms.RandomVerticalFlip(p=0.5),transforms.RandomRotation(45)]

transform_train = transforms.Compose([
        transforms.Resize((128,64)),
        transforms.RandomChoice(transform),
        # 转换成PyTorch能够计算和处理的Tensor数据类型的变量
        transforms.ToTensor(),
        transforms.Normalize((0.37169233, 0.38456926, 0.3438824), (0.20752552, 0.18884005, 0.18621244))
])

# 1,cifar10; 2, data_rs; 3, NWPU; 4, UCMerced_LandUse
transform_test = transforms.Compose([
        transforms.Resize((128,64)),
        transforms.ToTensor(),
        transforms.Normalize((0.37169233, 0.38456926, 0.3438824), (0.20752552, 0.18884005, 0.18621244))
])

####------------------------------import data list-------------------------------------------####

train_txt_path = './data/train.txt'
valid_txt_path = './data/test.txt'

print(train_txt_path, valid_txt_path)

# 构建MyDataset实例
valid_data = MyDataset(txt_path=train_txt_path, transform=transform_test)
train_data = MyDataset(txt_path=valid_txt_path, transform=transform_train)

# 数据加载
trainloader = DataLoader(dataset=train_data, batch_size=train_bs, shuffle=True)
testloader = DataLoader(dataset=valid_data, batch_size=valid_bs)

classes = ('pig0', 'pig1', 'pig2', 'pig3', 'pig4', 'pig5', 'pig6', 'pig7')

num_classes = len(classes)

# net definition
# 训练过程保存模型，训练过程中断后加载先前保存的模型，这里相当于记录训练次数
start_epoch = 0

# 总共训练次数
total_epoch = 40

net = Net(num_classes=num_classes)

# 这个参数主要是用来设置是否从断点处继续训练
if args.resume:
    # isfile(args.resume):里面的（）放入自己的最好模型的具体位置
    assert os.path.isfile("./checkpoint/ckpt.t7"), "Error: no checkpoint file found!"
    print('Loading from checkpoint/ckpt.t7')
    # checkpoint = torch.load(具体位置)
    checkpoint = torch.load("./checkpoint/ckpt.t7")
    # import ipdb; ipdb.set_trace()
    net_dict = checkpoint['net_dict']
    net.load_state_dict(net_dict)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
else:
    print('new model')

net.to(device)

# loss and optimizer
# 网络定义好之后，还要定义模型的损失函数和对参数进行优化的优化函数
# 优化函数使用的SGD，损失函数使用的是交叉熵
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), args.lr, momentum=0.9, weight_decay=5e-4)
best_acc = 0.

# train function for each epoch
def train(epoch):
    print("\n Epoch : %d"%(epoch+1))
    net.train()
    training_loss = 0.
    train_loss = 0.
    correct = 0
    total = 0
    interval = args.interval
    start = time.time()
    for idx, (inputs, labels) in enumerate(trainloader):
        # forward
        inputs,labels = inputs.to(device),labels.to(device)
        outputs = net(inputs)
        loss = criterion(outputs, labels)

        # backward
        # 引入优化算法，要调用optimizer.zero_grad()完成对模型参数梯度的归0
        # 如果没有optimizer.zero_grad()，梯度会累加在一起，结果不收敛
        optimizer.zero_grad()
        # 根据计算图自动计算每个节点的梯度值，并根据需要进行保留
        # backward主要是模型的反向传播中的自动梯度计算，在网络定义中的forward是模型前向传播中的矩阵计算
        loss.backward()
        # optimizer.step()作用是使用计算得到的梯度值对各个节点参数进行梯度更新
        optimizer.step()

        # accumurating
        # 注意显存
        training_loss += loss.item()
        train_loss += loss.item()
        correct += outputs.max(dim=1)[1].eq(labels).sum().item()
        total += labels.size(0)

        # print
        # 每个一段时间将损失的值打印出来看，确定模型误差越来越小
        # Variable会放在计算图中，然后进行前向传播，反向传播，自动求导
        # 可以通过data取出Variable中的tensor数值
        # 如果要打印，还可以用loss.data[0]
        if (idx+1)%interval == 0:
            end = time.time()
            print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
                100.*(idx+1)/len(trainloader), end-start, training_loss/interval, correct, total, 100.*correct/total
            ))
            training_loss = 0.
            start = time.time()
    
    return train_loss/len(trainloader), 1.- correct/total

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0.
    correct = 0
    total = 0
    start = time.time()

    # #不需要bp的forward, 注意model.eval() 不等于 torch.no_grad()
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            correct += outputs.max(dim=1)[1].eq(labels).sum().item()
            total += labels.size(0)
        
        print("Testing ...")
        end = time.time()
        print("[progress:{:.1f}%]time:{:.2f}s Loss:{:.5f} Correct:{}/{} Acc:{:.3f}%".format(
                100.*(idx+1)/len(testloader), end-start, test_loss/len(testloader), correct, total, 100.*correct/total
            ))

    # saving checkpoint
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
        print("Saving parameters to checkpoint/ckpt.t7")
        checkpoint = {
            'net_dict':net.state_dict(),
            'acc':acc,
            'epoch':epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(checkpoint, './checkpoint/ckpt.t7')

    return test_loss/len(testloader), 1.- correct/total

# plot figure
x_epoch = []
record = {'train_loss':[], 'train_err':[], 'test_loss':[], 'test_err':[]}
fig = plt.figure()
ax0 = fig.add_subplot(121, title="loss")
ax1 = fig.add_subplot(122, title="top1err")

# 画出损失曲线图
def draw_curve(epoch, train_loss, train_err, test_loss, test_err):
    global record
    record['train_loss'].append(train_loss)
    record['train_err'].append(train_err)
    record['test_loss'].append(test_loss)
    record['test_err'].append(test_err)

    x_epoch.append(epoch)
    ax0.plot(x_epoch, record['train_loss'], 'bo-', label='train')
    ax0.plot(x_epoch, record['test_loss'], 'ro-', label='val')
    ax1.plot(x_epoch, record['train_err'], 'bo-', label='train')
    ax1.plot(x_epoch, record['test_err'], 'ro-', label='val')
    if epoch == 0:
        ax0.legend()
        ax1.legend()
    fig.savefig("train.jpg")

# lr decay，学习率衰减方法
# 线性衰减。例如：每过20个epochs学习率减半
# 指数衰减。例如：每过20个epochs将学习率乘以0.1
def lr_decay():
    global optimizer
    for params in optimizer.param_groups:
        params['lr'] *= 0.1
        lr = params['lr']
        print("Learning rate adjusted to {}".format(lr))

def main():
    for epoch in range(start_epoch, start_epoch+total_epoch):
        train_loss, train_err = train(epoch)
        test_loss, test_err = test(epoch)
        draw_curve(epoch, train_loss, train_err, test_loss, test_err)
        if (epoch+1)%20==0:
            lr_decay()

if __name__ == '__main__':
    main()
