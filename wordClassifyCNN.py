import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as functional
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from visdom import Visdom
from pytorchtools import EarlyStopping
from torch.optim import lr_scheduler

class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1,6,kernel_size=5,stride=1)#输入图像通道数1,卷积产生的通道数6
        self.conv2 = nn.Conv2d(6,16,kernel_size=5,stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm2d(num_features=6)
        self.bn2 = nn.BatchNorm2d(num_features=16)
        self.bn3 = nn.BatchNorm1d(num_features=120)
        self.fc1 = nn.Linear(256,120)#全连接层
        self.fc3 = nn.Linear(120,12)


        # self.fc2 = nn.Linear(120,84)
        # self.fc3 = nn.Linear(84,12)

    def forward(self,x):
        """override了nn.Module的forward。在__call__中被调用"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = functional.relu(x)
        x = functional.max_pool2d(x,(2, 2))

        x = self.conv2(x)
        x = self.bn2(x)
        x = functional.relu(x)
        x = functional.max_pool2d(x,(2, 2))
        x = self.dropout1(x)
        x = torch.flatten(x,1) #把x展平成(batchSize,256)的tensor
        x = self.fc1(x)
        x = self.bn3(x)
        x = functional.relu(x)
        x = self.dropout2(x)
        # x = self.fc2(x)
        # x = functional.relu(x)
        x = self.fc3(x)
        return functional.log_softmax(x,dim=1)

def train(model,epoch,trainLoader,optimizer):
    model.train()#打开训练模式
    for batchIndex,(data,label) in enumerate(trainLoader):
        data,label = Variable(data),Variable(label)
        output = model(data) #计算前向传播
        optimizer.zero_grad() #上一次的梯度记录被清空
        loss = functional.nll_loss(output,label)
        loss.backward() #自动计算梯度
        optimizer.step() #更新参数
        if(batchIndex%20==0):#每20轮输出结果
            print('epoch:%s,batch:%s,Loss:%.6f'%(epoch,batchIndex,loss.item()))
    return loss.item()

def valuate(model,testLoader):
    model.eval()#切换evaluation模式
    loss,accurate = 0,0
    with torch.no_grad():#上下文管理器，不追踪梯度
        for data,label in testLoader:
            output = model(data)
            predict = output.argmax(dim=1) #argmax按行得到预测出的类别的索引
            accurate += sum(label==predict).item()
            loss += functional.nll_loss(output, label, reduction='sum').item()  # 累加loss
        loss = loss/len(testLoader.dataset)
        accurate = accurate/len(testLoader.dataset)
        print('test loss:%s , accuracy:%s %%'%(loss,accurate*100))
    return loss,accurate

def loadData():
    DATAPATH = "./train"
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), #彩色图像转灰度图像
        transforms.ToTensor(),#将PIL.Image/numpy.ndarray数据转化为torch.FloadTensor，归一化
    ])
    allData = ImageFolder(DATAPATH,transform)
    trainSize = int(0.8 * len(allData))
    testSize = len(allData) - trainSize
    torch.manual_seed(0)
    trainSubSet, testSubSet= torch.utils.data.random_split(dataset= allData, lengths=[trainSize,testSize])
    trainLoader = torch.utils.data.DataLoader(trainSubSet, batch_size=16, shuffle=True, num_workers=0)
    testLoader  = torch.utils.data.DataLoader(testSubSet,  batch_size=128, shuffle=True, num_workers=0)
    return trainLoader,testLoader

def trainMode():
    model = CNN()
    viz = Visdom() 
    viz.line([[2.5,2.5]], [0], win='loss', opts=dict(title='train & test loss', legend=['train', 'test']))#初始化显示loss曲线的窗口
    viz.line([0.], [0], win='accuracy', opts=dict(title='accuracy', legend=['accuracy']))#初始化显示准确率曲线的窗口
    trainLoader,testLoader = loadData()

    ##梯度下降的优化器
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,weight_decay=1e-5) #SGD with momentum，一阶动量
    # optimizer= optim.RMSprop(model.parameters(),lr=0.01,alpha=0.9) #引入二阶动量
    # optimizer=optim.Adam(model.parameters(),lr=0.01) #一阶动量+二阶动量
    ##学习率衰减（对于SGD）
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7) #等间隔调整
    # scheduler = lr_scheduler.MultiStepLR(optimizer,milestones=[5,10],gamma = 0.8) #设定的间隔调整学习率
    # scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9) #指数衰减
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max = 20) #余弦退火调整
    # scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08) #自适应调整
    early_stopping = EarlyStopping(patience=5, verbose=True)

    for epoch in range(1, 1000+1):
        trainLoss = train(model,epoch,trainLoader,optimizer)
        testLoss,accuracy = valuate(model,testLoader)
        scheduler.step()
        viz.line([[trainLoss, testLoss]], [epoch], win='loss', update='append')
        viz.line([accuracy], [epoch], win='accuracy', update='append')
        early_stopping(1-accuracy, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

def testMode(path):
    model = CNN()
    # 将模型参数加载到新模型中
    state_dict = torch.load('checkpoint.pt')
    model.load_state_dict(state_dict)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), #彩色图像转灰度图像
        transforms.ToTensor(),#将PIL.Image/numpy.ndarray数据转化为torch.FloadTensor，归一化
    ])
    testData = ImageFolder(path,transform)
    testLoader  = torch.utils.data.DataLoader(testData,  batch_size=4, shuffle=True, num_workers=0)
    valuate(model,testLoader)

if __name__=='__main__':
    mode = input('测试输入test：')
    if mode == 'test':
        print('test start!')
        path = input('输入测试图片集路径：')
        testMode(path)
    else:
        trainMode()
    