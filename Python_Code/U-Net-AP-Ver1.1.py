# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 20:28:10 2022

@author: Sanctified
"""

import pickle

from sklearn.cluster import MiniBatchKMeans
import os
from PIL import Image
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from tqdm.notebook import tqdm

# GPU 사용이 가능할 경우, GPU를 사용할 수 있게 함.
device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
print(device)

# 자신의 폴더 경로에 맞게 재지정해주세요.
root_path = 'D:/Medical Image processing/AP'

data_dir = os.path.join(root_path,'Dataset/DICOM/CT Combine/')

# data_dir의 경로(문자열)와 train(문자열)을 결합해서 train_dir(train 폴더의 경로)에 저장합니다.
train_dir = os.path.join(data_dir, "L3train")  # l3만

# data_dir의 경로(문자열)와 val(문자열)을 결합해서 val_dir(val 폴더의 경로)에 저장합니다.
val_dir = os.path.join(data_dir, "L3val")  # l3만
val_dir_tst = os.path.join(data_dir, "L3tst")  # TEST IMAGE ONLY (의사용)

# train_dir 경로에 있는 모든 파일을 리스트의 형태로 불러와서 train_fns에 저장합니다.
train_fns = os.listdir(train_dir)

# val_dir 경로에 있는 모든 파일을 리스트의 형태로 불러와서 val_fns에 저장합니다.
val_fns = os.listdir(val_dir)

# print(len(train_fns), len(val_fns))


# label model 불러오기 위한 경로 코드 
label_model_path = os.path.join(root_path, '\코드\label_model.spydata')

# # train_dir(문자열)와 train_fns[0](문자열)의 경로를 결합하여 sample_image_fp(샘플 이미지의 경로)에 저장합니다.
sample_image_fp = os.path.join(train_dir, train_fns[59])

# PIL 라이브러리의 Image 모듈을 사용하여, sample_image_fp를 불러옵니다.

path = os.path.join('r', sample_image_fp)
sample_image = Image.open(path)
sample_image = Image.open(path).convert("RGB")

plt.imshow(sample_image)
plt.show()

num_items = len(train_fns)

# # 0~255 사이의 숫자를 3*num_items번 랜덤하게 뽑기
color_array = np.random.choice(range(256), 3*num_items).reshape(-1, 3)
print(color_array.shape)

num_classes = 2

# K-means clustering 알고리즘을 사용하여 label_model에 저장합니다.
label_model = KMeans(n_clusters=10, random_state=0)
label_model = MiniBatchKMeans(
    init="k-means++",
    n_clusters=10,
    batch_size=16,
    n_init=20,
    max_no_improvement=10,
    verbose=0,
)
label_model.fit(color_array)
saved_model = pickle.dumps(label_model)

np.save('saved_label_model.npy', saved_model)



# 이전에 샘플이미지에서 볼 수 있듯이, original image와 labeled image가 연결되어 있는데 이를 분리해줍니다.

# def split_image(image):
#    image = np.array(image)

#    # 이미지의 크기가 256 x 512 였는데 이를 original image와 labeled image로 분리하기 위해 리스트로 슬라이싱 합니다.
#    # 그리고 분리된 이미지를 각각 cityscape(= original image)와 label(= labeled image)에 저장합니다.
#    APimage, label = image[:, :256, :], image[:, 256:, :]
#    return APimage, label


# # 바로 이전 코드에서 정의한 split_image() 함수를 이용하여 sample_image를 분리한 후, cityscape과 label에 각각 저장합니다.
# APimage, label = split_image(sample_image)

# label_class = label_model.predict(label.reshape(-1, 3)).reshape(256, 256)
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# axes[0].imshow(APimage)
# axes[1].imshow(label)
# axes[2].imshow(label_class)

# plt.show()

# label_class = label_model.predict(label.reshape(-1, 3)).reshape(256, 256)
# # label_class = label_model.predict(label.reshape(-1, 1)).reshape(256, 256)
# print(label.shape)
# print(label.reshape(-1, 3).shape)
# print(label_model.predict(label.reshape(-1, 3)).shape)
# print(label_class.shape)



class APDataset(Dataset):

  def __init__(self, image_dir, label_model):
    self.image_dir = image_dir
    self.image_fns = os.listdir(image_dir)
    self.label_model = label_model

  def __len__(self):
    return len(self.image_fns)

  def __getitem__(self, index):
    image_fn = self.image_fns[index]
    image_fp = os.path.join(self.image_dir, image_fn)
    image = Image.open(image_fp)
    image = np.array(image)
    APimage, label = self.split_image(image)
    label_class = self.label_model.predict(
        label.reshape(-1, 3)).reshape(256, 256)
    label_class = torch.Tensor(label_class).long()
    APimage = self.transform(APimage)
    return APimage, label_class

  def split_image(self, image):
    image = np.array(image)
    APimage, label = image[:, :256, :], image[:, 256:, :]
    return APimage, label

  def transform(self, image):
    transform_ops = transforms.Compose([
      			transforms.ToTensor()])
    return transform_ops(image)

saved_model = np.load('saved_label_model.npy')
label_model=pickle.loads(saved_model)
dataset = APDataset(train_dir, label_model)
print(len(dataset))

# APimage, label_class = dataset[0]
# print(APimage.shape)
# print(label_class.shape)


# class UNet(nn.Module):

#     def __init__(self, num_classes):
#         super(UNet, self).__init__()
#         self.num_classes = num_classes
#         self.contracting_11 = self.conv_block(in_channels=3, out_channels=64)
#         self.contracting_12 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.contracting_21 = self.conv_block(in_channels=64, out_channels=128)
#         self.contracting_22 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.contracting_31 = self.conv_block(
#             in_channels=128, out_channels=256)
#         self.contracting_32 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.contracting_41 = self.conv_block(
#             in_channels=256, out_channels=512)
#         self.contracting_42 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.middle = self.conv_block(in_channels=512, out_channels=1024)
#         self.expansive_11 = nn.ConvTranspose2d(
#             in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.expansive_12 = self.conv_block(in_channels=1024, out_channels=512)
#         self.expansive_21 = nn.ConvTranspose2d(
#             in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.expansive_22 = self.conv_block(in_channels=512, out_channels=256)
#         self.expansive_31 = nn.ConvTranspose2d(
#             in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.expansive_32 = self.conv_block(in_channels=256, out_channels=128)
#         self.expansive_41 = nn.ConvTranspose2d(
#             in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
#         self.expansive_42 = self.conv_block(in_channels=128, out_channels=64)
#         self.output = nn.Conv2d(
#             in_channels=64, out_channels=num_classes, kernel_size=3, stride=1, padding=1)

#     def conv_block(self, in_channels, out_channels):
#         block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
#                                     nn.ReLU(),
#                                     nn.BatchNorm2d(num_features=out_channels),
#                                     nn.Conv2d(
#                                         in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
#                                     nn.ReLU(),
#                                     nn.BatchNorm2d(num_features=out_channels))
#         return block

#     def forward(self, X):
#         contracting_11_out = self.contracting_11(X)  # [-1, 64, 256, 256]
#         contracting_12_out = self.contracting_12(
#             contracting_11_out)  # [-1, 64, 128, 128]
#         contracting_21_out = self.contracting_21(
#             contracting_12_out)  # [-1, 128, 128, 128]
#         contracting_22_out = self.contracting_22(
#             contracting_21_out)  # [-1, 128, 64, 64]
#         contracting_31_out = self.contracting_31(
#             contracting_22_out)  # [-1, 256, 64, 64]
#         contracting_32_out = self.contracting_32(
#             contracting_31_out)  # [-1, 256, 32, 32]
#         contracting_41_out = self.contracting_41(
#             contracting_32_out)  # [-1, 512, 32, 32]
#         contracting_42_out = self.contracting_42(
#             contracting_41_out)  # [-1, 512, 16, 16]
#         middle_out = self.middle(contracting_42_out)  # [-1, 1024, 16, 16]
#         expansive_11_out = self.expansive_11(middle_out)  # [-1, 512, 32, 32]
#         expansive_12_out = self.expansive_12(torch.cat(
#             (expansive_11_out, contracting_41_out), dim=1))  # [-1, 1024, 32, 32] -> [-1, 512, 32, 32]
#         expansive_21_out = self.expansive_21(
#             expansive_12_out)  # [-1, 256, 64, 64]
#         expansive_22_out = self.expansive_22(torch.cat(
#             (expansive_21_out, contracting_31_out), dim=1))  # [-1, 512, 64, 64] -> [-1, 256, 64, 64]
#         expansive_31_out = self.expansive_31(
#             expansive_22_out)  # [-1, 128, 128, 128]
#         # [-1, 256, 128, 128] -> [-1, 128, 128, 128]
#         expansive_32_out = self.expansive_32(
#             torch.cat((expansive_31_out, contracting_21_out), dim=1))
#         expansive_41_out = self.expansive_41(
#             expansive_32_out)  # [-1, 64, 256, 256]
#         # [-1, 128, 256, 256] -> [-1, 64, 256, 256]
#         expansive_42_out = self.expansive_42(
#             torch.cat((expansive_41_out, contracting_11_out), dim=1))
#         # [-1, num_classes, 256, 256]
#         output_out = self.output(expansive_42_out)
#         return output_out


# model = UNet(num_classes=num_classes)

# data_loader = DataLoader(dataset, batch_size=4)
# print(len(dataset), len(data_loader))

# X, Y = iter(data_loader).next()
# print(X.shape)
# print(Y.shape)

# Y_pred = model(X)
# print(Y_pred.shape)


class UNet(nn.Module):

    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.num_classes = num_classes
        self.contracting_11 = self.conv_block(in_channels=3, out_channels=64)
        self.contracting_12 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_21 = self.conv_block(in_channels=64, out_channels=128)
        self.contracting_22 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_31 = self.conv_block(
            in_channels=128, out_channels=256)
        self.contracting_32 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.contracting_41 = self.conv_block(
            in_channels=256, out_channels=512)
        self.contracting_42 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.middle = self.conv_block(in_channels=512, out_channels=1024)
        self.expansive_11 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_12 = self.conv_block(in_channels=1024, out_channels=512)
        self.expansive_21 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_22 = self.conv_block(in_channels=512, out_channels=256)
        self.expansive_31 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_32 = self.conv_block(in_channels=256, out_channels=128)
        self.expansive_41 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.expansive_42 = self.conv_block(in_channels=128, out_channels=64)
        self.output = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
	# 1x1 convolution layer 추가
        self.output1 = nn.Conv2d(
            in_channels=64, out_channels=num_classes, kernel_size=1, stride=1, padding=1)

    def conv_block(self, in_channels, out_channels):
        block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=out_channels),
                                    nn.Conv2d(
                                        in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.BatchNorm2d(num_features=out_channels))
        return block

    def forward(self, X):
        contracting_11_out = self.contracting_11(X)  # [-1, 64, 256, 256]
        contracting_12_out = self.contracting_12(
            contracting_11_out)  # [-1, 64, 128, 128]
        contracting_21_out = self.contracting_21(
            contracting_12_out)  # [-1, 128, 128, 128]
        contracting_22_out = self.contracting_22(
            contracting_21_out)  # [-1, 128, 64, 64]
        contracting_31_out = self.contracting_31(
            contracting_22_out)  # [-1, 256, 64, 64]
        contracting_32_out = self.contracting_32(
            contracting_31_out)  # [-1, 256, 32, 32]
        contracting_41_out = self.contracting_41(
            contracting_32_out)  # [-1, 512, 32, 32]
        contracting_42_out = self.contracting_42(
            contracting_41_out)  # [-1, 512, 16, 16]
        middle_out = self.middle(contracting_42_out)  # [-1, 1024, 16, 16]
        expansive_11_out = self.expansive_11(middle_out)  # [-1, 512, 32, 32]
        expansive_12_out = self.expansive_12(torch.cat(
            (expansive_11_out, contracting_41_out), dim=1))  # [-1, 1024, 32, 32] -> [-1, 512, 32, 32]
        expansive_21_out = self.expansive_21(
            expansive_12_out)  # [-1, 256, 64, 64]
        expansive_22_out = self.expansive_22(torch.cat(
            (expansive_21_out, contracting_31_out), dim=1))  # [-1, 512, 64, 64] -> [-1, 256, 64, 64]
        expansive_31_out = self.expansive_31(
            expansive_22_out)  # [-1, 128, 128, 128]
        # [-1, 256, 128, 128] -> [-1, 128, 128, 128]
        expansive_32_out = self.expansive_32(
            torch.cat((expansive_31_out, contracting_21_out), dim=1))
        expansive_41_out = self.expansive_41(
            expansive_32_out)  # [-1, 64, 256, 256]
        # [-1, 128, 256, 256] -> [-1, 64, 256, 256]
        expansive_42_out = self.expansive_42(
            torch.cat((expansive_41_out, contracting_11_out), dim=1))
        # [-1, 64, 256, 256] -> [-1, 64, 256, 256]
        output_out = self.output(expansive_42_out)
        output_out1 = self.output(output_out)  # [-1, num_classes, 256, 256]

        return output_out1


batch_size = 8  # num_items/2

epochs = 100
lr = 0.0001

dataset = APDataset(train_dir, label_model)
data_loader = DataLoader(dataset, batch_size=batch_size)

LoadorNot = 1
if LoadorNot == 1:
    model = UNet(num_classes=num_classes).to(device)
    
    from tensorflow.keras.utils import plot_model
    # model.summary()
    # plot_model(model, to_file='model.png', show_shapes=True)
    
    # 손실함수 정의
    criterion = nn.CrossEntropyLoss()
    # Optimizer 정의
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    
    step_losses = []
    epoch_losses = []
    
    for epoch in tqdm(range(epochs)):
      epoch_loss = 0
    
      for X, Y in tqdm(data_loader, total=len(data_loader), leave=False):
        X, Y = X.to(device), Y.to(device)
        optimizer.zero_grad()
        Y_pred = model(X)
        loss = criterion(Y_pred, Y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        step_losses.append(loss.item())
    
      epoch_losses.append(epoch_loss/len(data_loader))
    
    print(len(epoch_losses))
    print(epoch_losses)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].plot(step_losses)
    axes[1].plot(epoch_losses)
    
    plt.show()
    
    model_name = "UNet.pth"
    torch.save(model.state_dict(), data_dir + model_name)
elif LoadorNot == 0:
    model_name = "UNet.pth"
    model_path = data_dir + model_name
    model_ = UNet(num_classes=num_classes).to(device)
    model_.load_state_dict(torch.load(model_path))

############
test_batch_size = 1
dataset_test = APDataset(val_dir, label_model)
data_loader_test = DataLoader(dataset_test, batch_size=test_batch_size)

# train_dir(문자열)와 train_fns[0](문자열)의 경로를 결합하여 sample_image_fp(샘플 이미지의 경로)에 저장합니다.
sample_image_fp = os.path.join(val_dir, val_fns[20])
# PIL 라이브러리의 Image 모듈을 사용하여, sample_image_fp를 불러옵니다.
path = os.path.join('r', sample_image_fp)
# sample_image = Image.open(path)
sample_image = Image.open(path).convert("RGB")
plt.imshow(sample_image)
plt.show()

"""
X, Y = next(iter(data_loader))
X, Y = X.to(device), Y.to(device)
Y_pred = model_(X)
print(Y_pred.shape)
Y_pred1 = torch.argmax(Y_pred, dim=1)
print(Y_pred1.shape)
"""

fig, axes = plt.subplots(test_batch_size, 3, figsize=(3*5, test_batch_size*5))
iou_scores = []
epochs = 16
lr = 0.001
step_losses = []
epoch_losses = []

from PIL import Image
import matplotlib.image

OriginalImagedir = 'D:\Medical Image processing\AP\Dataset\DICOM\CT Unet\Original/'
LabelImagedir = 'D:\Medical Image processing\AP\Dataset\DICOM\CT Unet/Label/'
PredictImagedir = 'D:\Medical Image processing\AP\Dataset\DICOM\CT Unet/Predict/'

inverse_transform = transforms.Compose([])

Pixeltomm = (0.7167/2) * (0.7167/2)
iou_scores=[]

XAll=[]
YAll=[]
Aplocation_list = []
label_class_list =[]
label_class_predicted_list=[]
Label_Fat = np.empty((0,0),int)
Label_predicted_Fat = np.empty((0,0),int)
Fat_mm_list = []
Fat_predicted_mm_list = []
Label_Fat_list = []
Label_predicted_Fat_list = []
# for i in range(test_batch_size):
for i, data in enumerate (data_loader_test, 0):
    cnt = 1;
    
    X, Y = data
    # X, Y = next(iter(data_loader_test))
    X, Y = X.to(device), Y.to(device)
   # print(X.shape)
   # print(Y.shape)
    Y_pred = model(X)
   # print(Y_pred.shape)
    Y_pred1 = torch.argmax(Y_pred, dim=1)          
   # print(Y_pred1.shape)
    label_class_predicted = Y_pred1[0].cpu().detach().numpy()
    
    X = torch.squeeze(X[0])
    Y = torch.squeeze(Y[0])
    
    Aplocation = inverse_transform(X).permute(1, 2, 0).cpu().detach().numpy()
    label_class = Y.cpu().detach().numpy()   
    
    plt.show()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(Aplocation)
    axes[1].imshow(label_class)
    axes[2].imshow(label_class_predicted)
            
    # IOU score
    # intersection = np.logical_and(label_class, label_class_predicted)
    # union = np.logical_or(label_class, label_class_predicted)
    # iou_score = np.sum(intersection) / np.sum(union)        
    cnt_iou_intersection = 0;
    for j in range(label_class_predicted.shape[0]):
        for j in range(label_class_predicted.shape[1]):
            if label_class[i][j] == label_class_predicted[i][j]:
                cnt_iou_intersection +=1
        
    intersection = np.logical_and(label_class, label_class_predicted)
    union = 256*256
    iou_score = cnt_iou_intersection /union
    Aplocation_list.append(Aplocation)
    label_class_list.append(label_class)
    label_class_predicted_list.append(label_class_predicted)
    iou_scores.append(iou_score)
    
    
    cntF = 0;
    cntM = 0;
    unique = np.unique(label_class)
    for j in range(label_class.shape[0]):
        for j in range(label_class.shape[1]):
            if label_class[i,j] == 4:
                cntF += 1
                
    Label_Fat = cntF
    Fat_mm = cntF * Pixeltomm    
    Label_Fat_list.append(Label_Fat)
    Fat_mm_list.append(Fat_mm)
    
    cntF = 0;
    cntM = 0;
    unique = np.unique(label_class_predicted)
    for j in range(label_class_predicted.shape[0]):
        for j in range(label_class_predicted.shape[1]):
            if label_class_predicted[i,j] == 4:
                cntF += 1
                
    Label_predicted_Fat = cntF
    Fat_predicted_mm = cntF * Pixeltomm
    Label_predicted_Fat_list.append(Label_predicted_Fat)
    Fat_predicted_mm_list.append(Fat_predicted_mm)
    
    matplotlib.image.imsave(OriginalImagedir+"AP_Original_"+str(i)+".png", Aplocation)
    matplotlib.image.imsave(LabelImagedir+"AP_Label_"+str(i)+".png", label_class)
    matplotlib.image.imsave(PredictImagedir+"AP_Predict_"+str(i)+".png", label_class_predicted)
    
    cnt = cnt + 1

np.save('Fat_predicted_mm.npy', Fat_predicted_mm)
np.save('Fat_predicted_mm_list.npy', Fat_predicted_mm_list)

##im = Image.fromarray(Aplocation)
##im.save("AP_Original")

i=10
plt.show()
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(Aplocation_list[i])
axes[1].imshow(label_class)
axes[2].imshow(label_class_predicted_list[i])
                

matplotlib.image.imsave('AP_Original.png', Aplocation_list[0])

"""
for i in range(test_batch_size):
    
    landscape = inverse_transform(X[i]).permute(1, 2, 0).cpu().detach().numpy()
    label_class = Y[i].cpu().detach().numpy()
    label_class_predicted = Y_pred1[i].cpu().detach().numpy()
    
    # IOU score
    intersection = np.logical_and(label_class, label_class_predicted)
    union = np.logical_or(label_class, label_class_predicted)
    iou_score = np.sum(intersection) / np.sum(union)
    iou_scores.append(iou_score)

    axes[i, 0].imshow(landscape)
    axes[i, 0].set_title("Landscape")
    axes[i, 1].imshow(label_class)
    axes[i, 1].set_title("Label Class")
    axes[i, 2].imshow(label_class_predicted)
    axes[i, 2].set_title("Label Class - Predicted")
    
plt.show()
"""

# dim = 1 -> dim = 0으로 변경해봄
Y_pred = torch.argmax(Y_pred, dim=0)


print(sum(iou_scores) / len(iou_scores))

####
np.save('Aplocation_list.npy', Aplocation_list)
np.save('label_class_list.npy', label_class_list)
np.save('label_class_predicted_list.npy', label_class_predicted_list)
np.save('iou_scores.npy', iou_scores)

plt.show()
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(Aplocation)
axes[1].imshow(label_class)
axes[2].imshow(label_class_predicted)
