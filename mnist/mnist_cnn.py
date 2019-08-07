# import torch
# import torch.nn as nn
# from torch.autograd import Variable
# import torch.utils.data as Data
# import torchvision
# import matplotlib as plt
# import numpy as np
#
# EPOCH = 1
# BATCH_SIZE = 50
# LR = 0.001
# DOWNLOAD_MNIST = False
#
# train_data = torchvision.datasets.MNIST(
#     root = './mnistDataset',
#     train = True,
#     transform = torchvision.transforms.ToTensor(),
#     download = DOWNLOAD_MNIST
# )
#
# #plot one example
# # print(train_data.train_data.size())
# # print(train_data.train_labels.size())
# # plt.imshow(train_data, train_data[0].numpy(), cmap = 'gray')
# # plt.title('%d' % train_data.train_labels[0])
# # plt.show()
#
# # 数据加载器
# train_loader = Data.DataLoader(
#         dataset = train_data,
#         batch_size = BATCH_SIZE,
#         shuffle = True,
#         num_workers = 0
# )
#
# #测试集
# test_data = torchvision.datasets.MNIST(
#     root = './mnistDataset',
#     train = False,  #提取的是test data
# )
#
# #in CPU
# # test_x = Variable(torch.unsqueeze(test_data.test_data, dim = 1), volatile = True).type(torch.FloatTensor)[:2000]/255.
# # test_y = test_data.test_labels[:2000]
#
# #in GPU
# test_x = Variable(torch.unsqueeze(test_data.test_data, dim = 1), volatile = True).type(torch.FloatTensor)[:2000].cuda()/255.
# test_y = test_data.test_labels[:2000].cuda()
#
# #建立神经网络
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(
#                 in_channels = 1,  #进去的深度
#                 out_channels = 16,   #输出的深度
#                 kernel_size = 5,
#                 stride = 1,
#                 #为了保证后面输出的图片的长宽不变
#                 padding = 2,  #if stride = 1, padding = (kernel_size-1)/2
#             ),  #卷积的作用：提取特征  (1, 28, 28) -> (16, 28, 28)
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size = 2) #(16, 28, 28)->(16, 14, 14)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(16, 32, 5, 1, 2), #(16, 14, 14) -> (32, 14, 14)
#             nn.ReLU(),
#             nn.MaxPool2d(2)  #(32, 14, 14) -> (32, 7, 7)
#         )
#         self.out = nn.Linear(32 * 7 * 7, 10)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)   #(bacth_size, 32, 7, 7)
#         x = x.view(x.size(0), -1)  # 展平  (batch_size, 32*7*7)
#         output = self.out(x)
#         return output
#
# cnn = CNN()
# cnn = cnn.cuda()
# optimizer = torch.optim.Adam(cnn.parameters(), lr = LR)
# loss_func = nn.CrossEntropyLoss()
#
# #训练
# for epoch in range(EPOCH):
#     for step, (x, y) in enumerate(train_loader):
#         # in CPU
#         # v_x = Variable(x)
#         # v_y = Variable(y)
#
#         # train cnn in GPU
#         v_x = Variable(x).cuda()
#         v_y = Variable(y).cuda()
#
#         # print('v_x size： ', end = '')
#         # print(v_x.size())
#
#         output = cnn(v_x)
#         loss = loss_func(output, v_y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
#         if(step % 50 == 0):
#             # print("tets_y size：", end = '')
#             # print(test_y.size())
#             test_output = cnn(test_x)
#             #in CPU
#             #pred_y = torch.max(test_output, dim = 1)[1].data.squeeze()
#             #in GPU
#             pred_y = torch.max(test_output, dim = 1)[1].cuda().data.squeeze()
#             accuracy = float((pred_y.cpu() == test_y.cpu()).numpy().astype(int).sum()) / test_y.size(0)
#             print("Epoch: ", epoch, '| train loss: %.4f' % loss.cpu().data.numpy(), '| accuacy: %.2f' % accuracy)
#
# # 训练完了之后要保存
# torch.save(cnn, 'cnn_mnist.pkl')
#
#
# test_output = cnn(test_x[:10])
# #in CPU
# #pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
# #in GPU
# pred_y = torch.max(test_output, 1)[1].cpu().data.numpy().squeeze()
# print(pred_y, 'predicton number')
# print(test_y[:10].cpu().numpy(), "real number")
#



# 构建LeNet深度神经网络，实现手写体（kaggle）的识别

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np
import csv

# 读取数据集
def readTrainingDataset():
    ifile = open('digit-recognizer/train.csv')
    ifile.readline()  #表头
    lines = ifile.readlines()
    trainingDataset, trainingLabel = [], []
    # index = 0
    for line in lines:
        # if(index == 200):
        #     break;
        # index += 1
        #label = [0 for i in range(10)]
        line_list = line.split(',')
        #label[int(line_list[0])] = 1
        trainingLabel.append([int(line_list[0])])
        picture = [np.array(line_list[1:]).reshape(28, -1)]
        trainingDataset.append(picture)
    trainingLabel = np.array(trainingLabel).astype(float)
    trainingDataset = np.array(trainingDataset).astype(float)
    # #m, n = trainingDataset.shape
    # #trainingDataset.reshape(200, 1, 28, -1)
    # print(trainingDataset.shape)
    # print(type(trainingDataset))
    # print(trainingDataset[0])
    # print(trainingDataset.ndim)
    ifile.close()

    return trainingDataset, trainingLabel

#构建LeNet
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = 1,
                out_channels = 6,
                kernel_size = 5,
                stride = 1,
                #no padding
                padding = 0
            ),
            nn.ReLU(),  #使用relu，原始使用sigmoid
            nn.MaxPool2d(
                kernel_size = 2,
                stride = 2,
            ),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels = 6,
                out_channels = 16,
                kernel_size = 5,
                stride = 1,
                #no padding
                padding = 0
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size = 2,
                stride = 2,
            )
        )
        self.full_connection1 = nn.Conv2d(
            in_channels = 16,
            out_channels = 120,
            kernel_size = 4,
            stride = 1,
            padding = 0
        )
        self.full_connection2 = nn.Sequential(
            nn.Linear(in_features = 120, out_features = 84),
            nn.ReLU()
        )
        self.predict = nn.Linear(84, 10)

    def forward(self, x):
        #print(x.size())
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.full_connection1(x)
        x = x.view(x.size(0), -1)   # (batch_size, 120)
        x = self.full_connection2(x)
        x = x.view(x.size(0), -1)   # (batch_size, 84)
        output = self.predict(x)  #这里没有相信softmax

        return output

#训练cnn网络
def train(trainingDataset, trainingLabel):
    EPOCH = 2
    BATCH_SIZE = 1000
    LR = 0.01

    cnn = LeNet()
    #数据加载器
    trainingDataset = torch.from_numpy(trainingDataset).type(torch.FloatTensor)
    trainingLabel = torch.from_numpy(trainingLabel).type(torch.LongTensor)
    torch_dataset = Data.TensorDataset(trainingDataset, trainingLabel)
    data_loader = Data.DataLoader(
            dataset = torch_dataset,
            batch_size = BATCH_SIZE,
            shuffle = True,
            num_workers = 0
    )

    #优化器
    optimizer = torch.optim.Adam(cnn.parameters(), lr = LR)
    #学习率下降规则器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.8)

    #损失函数 - 交叉熵损失
    # one-hot 中的 hot
    loss_func = nn.CrossEntropyLoss()  # torch.LongTensor
    #损失函数 - 平方损失
    #loss_func = nn.MSELoss()  # torch.FloatTensor

    #开始训练 - 过拟合
    print("trainging.....")
    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(data_loader):
            # print("batch_x")
            # print(batch_x.size())
            #predict = torch.softmax(cnn.forward(batch_x), dim = 1)
            predict = cnn.forward(batch_x)
            # print(predict.size())
            # print(batch_y.squeeze().size())
            loss = loss_func(predict, batch_y.squeeze())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            print("epoch: %d\t" % epoch,
                  "step: %d\t" % step,
                  "lr: %f" % optimizer.state_dict()['param_groups'][0]['lr'],
                  "loss: %.4f" % loss.data.numpy())

    #训练完成之后保存网络
    torch.save(cnn, 'digit-recognizer_cnn_CrossE.pkl')

    return cnn

#读取测试数据
def readTestDataset():
    ifile = open('digit-recognizer/test.csv')
    ifile.readline()
    lines = ifile.readlines()
    dataset = []
    for line in lines:
        line_list = line.split(',')
        dataset.append([np.array(line_list).reshape(28, -1)])
    dataset = np.array(dataset).astype(float)
    ifile.close()
    return dataset

#测试
def test(testDataset, cnn):
    m = testDataset.shape[0]
    label_list = []
    for i in range(m):
        testData_tensor = torch.from_numpy(np.expand_dims(testDataset[i], axis = 0)).type(torch.FloatTensor)
        #print(testData_tensor.size())
        predict = cnn.forward(testData_tensor)
        predict_label = torch.max(predict, dim = 1)[1]
        label_list.append(int(predict_label.data.numpy().squeeze()))
    return label_list

#显示图片和预测标签
def show(label_list, testDataset, index):
    plt.imshow(testDataset[index].squeeze(), cmap = 'gray')
    plt.title('%d' % label_list[index])
    plt.show()

#写入文件
def result2csv(label_list):
    with open('digit-recognizer/submit_cnn_mnist_2.csv', 'w', encoding = 'utf-8', newline = '') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ImageId", "Label"])
        for i, label in enumerate(label_list):
            writer.writerow([i+1, label])

def main():
    #读取训练数据
    trainingDataset, trainingLabel = readTrainingDataset()

    #训练
    cnn = train(trainingDataset, trainingLabel)

    #读取CNN网络
    #cnn = LeNet()
    #cnn = torch.load('digit-recognizer_cnn_CrossE.pkl')

    #读取测试集合
    testDataset = readTestDataset()
    #测试
    label_list = test(testDataset, cnn)
    #show
    # index = 8
    # show(label_list, testDataset, index)

    #写入文件
    result2csv(label_list)


if __name__ == "__main__":
    main()




















