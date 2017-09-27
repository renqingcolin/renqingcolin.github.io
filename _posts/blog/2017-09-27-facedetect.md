---
layout: post
title: 人脸检测学习
categories: Blog
description: 人脸检测
keywords: 计算机视觉 人脸检测
---

# 目前人脸检测算法主要分为4大类：

>cascade based methods(基于)
><br>part based methods
><br>channel feature based methods
><br>neural network based methods

## cascade based methods
这个方法源自在2001年，Viola和Jones发表了经典的《Rapid Object Detection using a Boosted Cascade of Simple Features》，使用积分图方法快速计算harr_like特征，使用级联结构Adaboost分类器。

Opencv的人脸检测模块就是基于这个方法的。

## part based methods
主要是将人脸分为几个部分的集合，deformable part model（DPM）是其中经典的算法《Object detection with discriminatively trained part-based models. TPAMI, 2010》，其中比较好的实现是《Face detection without bells and whistles. In ECCV. 2014》

## channel feature based methods
这一块还没看，相关的论文有：

1. 《P. Dollar, Z. Tu, P. Perona, and S. Belongie. Integral channel features. In BMVC, 2009》

2. 《B. Yang, J. Yan, Z. Lei, and S. Z. Li. Aggregate channel features for multi-view face detection. CoRR, 2014》 

## neural network based methods

主要是使用卷积神经网络，知道的有：

1. faster rcnn  --已看
2. 《H. Li, Z. Lin, X. Shen, J. Brandt, and G. Hua. A convolutional neural network cascade for face detection. In CVPR, 2015》（网络与级联的结合） --已看
3. 《S. Yang, P. Luo, C. C. Loy, and X. Tang. From facial parts responses to face detection: A deep learning approach. In ICCV, 2015.》（part based与级联的结合） --已看

## 另外
如:

1. hog 《Histograms of Oriented Gradients for Human Detection》
2. tiny face 《Finding Tiny Faces》《WIDER FACE: A Face Detection Benchmark》

