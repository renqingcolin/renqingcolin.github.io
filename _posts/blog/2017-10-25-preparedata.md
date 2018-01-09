---
layout: post
title: 目标识别数据集准备
categories: Blog
description: 数据集 目标检测 
keywords: 视频目标检测
---

## voc2007数据集

### 下载
```
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
```
### 解压
```
tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar
tar xvf VOCdevkit_08-Jun-2007.tar
```

### 数据理解
#### ImageSets/Main文件夹
Main下存放的是图像物体识别的数据，总共分为20类。Main文件夹下包含了20个分类的_train.txt、_
val.txt和_trainval.txt。
>_train中存放的是训练使用的数据  
>_val中存放的是验证结果使用的数据  
>_trainval将上面两个进行了合并  

需要保证的是train和val两者没有交集，也就是训练数据和验证数据不能有重复。
这些txt中的内容，每行由图像的name和lable组成。其中1代表正例，-1代表负例，0代表包含多个目标的样本（包含正例）。

## 准备自己的数据集


## 修改配置文件
在models/pascal_voc/中对于三种网络