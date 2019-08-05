#使用nn测试mnist数据集

import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
import torch.utils.data as Data
import csv
#import matplotlib.pyplot as plt


def readTrainingData():
    ifile = open('digit-recognizer/train.csv')
    ifile.readline()  #表头
    lines = ifile.readlines()
    dataSet, labels = [], []
    # index = 0
    for line in lines:
        # if(index == 200):
        #     break;
        # index += 1
        line_list = line.split(',')
        label = [0 for i in range(10)]
        label[int(line_list[0])] = 1
        data = line_list[1:]
        dataSet.append(data)
        labels.append(label)
    dataSet = np.array(dataSet).astype(float)
    labels = np.array(labels).astype(float)

    ifile.close()
    #return dataSet.transpose(), labels.transpose()
    return dataSet, labels

#神经网络层结构
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        # x = Variable(x.data.t())
        # print('x')
        # print(x.data.numpy().shape)
        x = F.relu(self.hidden(x))
        x = F.softmax(self.predict(x), dim = 1)  #dim = 1?

        return x

#保存模型
def saveModel(net):
    torch.save(net, 'nn_mnist.pkl')  # 保存entire net

#训练神经网络
#数据集格式
'''
 n(feature)
 →→→
m↓
 ↓
'''
LR = 0.005
BATCH_SIZE = 100
EPOCH = 10
def train(trainingDataSet, triainingLabel):
    m, n = trainingDataSet.shape
    net = Net(n, 100, 10)
    trainData = torch.from_numpy(trainingDataSet).type(torch.FloatTensor)
    trainLabel = torch.from_numpy(triainingLabel).type(torch.FloatTensor)

    #交叉熵损失
    loss_func = torch.nn.MSELoss()
    #优化器
    optimizer = torch.optim.SGD(net.parameters(), lr = LR, momentum = 0.8)
    #数据加载器
    torch_dataset = Data.TensorDataset(trainData, trainLabel)
    loader = Data.DataLoader(
        dataset = torch_dataset,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers = 0
    )
    #开始迭代训练
    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(loader):
            vb_x = Variable(batch_x) #转置
            vb_y = Variable(batch_y)

            #print(vb_x.data.numpy().shape)
            #print(vb_y.data.numpy().shape)

            predict = net.forward(vb_x)
            #print("predict:")
            #print(predict.data.numpy().shape)
            loss = loss_func(predict, vb_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #打印信息
            print("iteration: %d   loss: %f" % (step, loss.data))
    #保存模型
    saveModel(net)

#读取测试数据集
def readTestData():
    ifile = open('digit-recognizer/test.csv')
    ifile.readline()#表头
    lines = ifile.readlines()
    testData = []
    for line in lines:
        testData.append(line.split(','))
    testData = np.array(testData).astype(float)
    #print(testData.shape)
    ifile.close()

    return testData

#测试网络
def test(net, testData):
    label_list = []
    for data in testData:
        data = Variable(torch.from_numpy(data.reshape(1, 784)).type(torch.FloatTensor))
        predict = net.forward(data)
        label = torch.max(predict, 1)[1]  #Variable
        label_list.append(int(label.data.numpy().squeeze()))

    return label_list

#结果写入到csv文件中
def result2csv(label_list):
    with open('digit-recognizer/submit_nn_mnist.csv', 'w', encoding = 'utf-8', newline = '') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ImageId", "Label"])
        for i, label in enumerate(label_list):
            writer.writerow([i+1, label])

def main():
    # #读取数据
    # trainingDataSet, trainingLabel = readTrainingData()
    # #print(trainingDataSet.shape)
    # #print(trainingLabel.shape)
    # print("readed training dataset")
    #
    # #训练
    # train(trainingDataSet, trainingLabel)
    #加载网络
    net = torch.load('nn_mnist.pkl')

    #读取测试数据集
    testData = readTestData()
    #测试
    label_list = test(net, testData)
    #将测试结果写入到csv文件
    result2csv(label_list)





if __name__ == "__main__":
    main()

