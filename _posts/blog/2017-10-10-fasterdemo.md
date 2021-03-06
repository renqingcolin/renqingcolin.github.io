---
layout: post
title: Faster rcnn
categories: Blog
description: Faster rcnn 
keywords: Faster rcnn
---

## 概念Fater rcnn理解

参考了以下文章：

>![Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)
><br>![CNN目标检测与分割（一）：Faster RCNN详解](http://blog.csdn.net/zy1034092330/article/details/62044941)

主要需要理解的概念：

>VGG、ZF网络结构，卷机网络结构
><br>RPN网络结构
><br>共享CONV5之前的卷积层

## Caffe的一些基本概念

由于Faster rcnn使用的深度模型框架是Caffe，所以对一下概念有些了解便于理解代码

### 网络结构文件
这个文件的后缀格式是.prototxt。就是编写你的网络有多少层，每一层有多少个特征图，输入、输出……。
以Faster rcnn中VGG16网络为例：
```
//这里只贴出了一点，具体见py-faster-rcnn/models/pascal_voc/VGG16/fast_rcnn/train.prototxt

name: "VGG_ILSVRC_16_layers"
layer {
  name: 'data' //data层
  type: 'Python'
  top: 'data'
  top: 'rois'
  top: 'labels'
  top: 'bbox_targets'
  top: 'bbox_inside_weights'
  top: 'bbox_outside_weights'
  python_param {
    module: 'roi_data_layer.layer'
    layer: 'RoIDataLayer'
    param_str: "'num_classes': 21"
  }
}
layer {
  name: "conv1_1" //卷积神经网络的第一层，卷积层
  type: "Convolution" //这层操作为卷积
  bottom: "data" //这一层的前一层是data层
  top: "conv1_1" 
  param {
    lr_mult: 0
    decay_mult: 0
  }
  param {
    lr_mult: 0
    decay_mult: 0
  }
  convolution_param { ##参数
    num_output: 64 //定义输出特征图个数 
    pad: 1 
    kernel_size: 3 //定义卷积核大小
  }
}
layer {
  name: "relu1_1"
  type: "ReLU"
  bottom: "conv1_1"
  top: "conv1_1"
}
```

### 网络求解文件
这个文件一般取名为：solver.prototxt，这个文件的后缀格式也是.prototxt。这个文件主要包含了一些求解网络，梯度下降参数、迭代次数等参数……，如下：
```
#该文件在py-faster-rcnn/models/pascal_voc/VGG16/fast_rcnn/solver.prototxt
train_net: "models/pascal_voc/VGG16/fast_rcnn/train.prototxt" 
#定义网络结构文件，也就是我们上一步编写的文件  
base_lr: 0.001 #学习率
lr_policy: "step" #梯度下降的相关优化策略 
gamma: 0.1
stepsize: 30000 #最大迭代次数
display: 20
average_loss: 100
# iter_size: 1
momentum: 0.9 #动量参数
weight_decay: 0.0005 #权重衰减系数
# We disable standard caffe solver snapshotting and implement our own snapshot
# function
snapshot: 0 #每迭代多少次，保存一次结果
# We still use the snapshot prefix, though
snapshot_prefix: "vgg16_fast_rcnn" #保存结果路径
#debug_info: true

```

### 数据格式
Caffe数据格式有lmdb和leveldb两种，转换过程可以见caffe的mnist例子，主要是在examples/mnist/convert_mnist_data.cpp

### Blobs
Caffe使用blobs结构来处理网络中时的数据和导数信息：blob是Caffe的标准数组结构，它提供了一个统一的内存接口。

## 依赖安装
### Caffe 安装
1.
```
git clone https://anonscm.debian.org/git/debian-science/packages/caffe-contrib.git
```
2. 安装python依赖
```
cd caffe/python
for req in $(cat requirements.txt); do pip install $req; done
```
3. 编译
```
make all -j32
make test
make runtest
make pycaffe -j32
```
4. 问题  
问题一
```
./include/caffe/util/hdf5.hpp:6:18: fatal error: hdf5.h: No such file or directory
解决办法：在Makefile.config找到以下行并添加hdf5部分
INCLUDE_DIRS := $(PYTHON_INCLUDE) /usr/local/include /usr/include/hdf5/serial 
LIBRARY_DIRS := $(PYTHON_LIB) /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu/hdf5/serial
```
问题二  
```
.build_release/tools/caffe: error while loading shared libraries: libcudart.so.9.0:
解决方法：
Run 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64' before running 'make all/test/runtest'
```
问题三：
```
LD -o .build_release/lib/libcaffe.so.1.0.0
/usr/bin/ld: cannot find -lcblas
/usr/bin/ld: cannot find -latlas
collect2: error: ld returned 1 exit status
解决方法：
cd /usr/lib64/atlas/
sudo ln -sv libsatlas.so.3.10 libcblas.so 
sudo ln -sv libsatlas.so.3.10 libatlas.so
```
## Faster Rcnn安装
从github上clone项目文件，注意：一定要在clone时加入--recursive参数，不然会很麻烦，也不要直接下载
```ls 
# Make sure to clone with --recursive
git clone --recursive https://github.com/rbgirshick/py-faster-rcnn.git
```
Cython模块的编译
```
cd py-faster-rcnn/lib
make
```
配置Makefile.config文件
```
cd py-faster-rcnn/caffe-fast-rcnn
cp Makefile.config.example Makefile.config
##配置完后..
make -j8 && make pycaffe
```
### 遇到的问题
#### 问题一
```
 File "/usr/local/lib/python2.7/dist-packages/matplotlib/__init__.py", line 123, in <module>
    from . import cbook
ImportError: cannot import name cbook
```
解决方法
1. Try to update matplotlib
```
//我原来的版本是2.1.0，后来改为2.0.2就好了
python -m pip install -U matplotlib
```
2. Try to reinstall matplotlib
```
python -m pip uninstall matplotlib
python -m pip install -U matplotlib==2.0.2
```
3. What does the following snippet prints to the console?
```
python -c "import matplotlib"
```

#### 问题二
```
ImportError: No module named easydict
```
解决方法
```
sudo pip install easydict 
```

### 参数的理解
```
# 数据路径
 'DATA_DIR': '/home/chengrenqing/py-faster-rcnn/data',
#缩放因子(从原图到feature map的坐标映射，可能会造成在原图上不同的box到了feature map坐标系上变得相同了 1/16)
 'DEDUP_BOXES': 0.0625,
# A small number that's used many times
 'EPS': 1e-14,
# Place outputs under an experiments directory
 'EXP_DIR': 'faster_rcnn_end2end',
 'GPU_ID': 1,
 'MATLAB': 'matlab',
# 模型路径
 'MODELS_DIR': '/home/chengrenqing/py-faster-rcnn/models/pascal_voc',
#所有network所用的像素均值设为相同
 'PIXEL_MEANS': array([[[ 102.9801,  115.9465,  122.7717]]]),
# For reproducibility
 'RNG_SEED': 3,
# 项目根路径
 'ROOT_DIR': '/home/chengrenqing/py-faster-rcnn',
----------train------------
 'TRAIN': {
            'PROPOSAL_METHOD': 'gt',
          #最短边Scale成600(can list multiple scales)
           'SCALES': [600],
          #最长边最大为1000
           'MAX_SIZE': 1000,

          #一个minibatch包含两张图片
           'IMS_PER_BATCH': 1,
          #Minibatch大小，即ROI的数量
           'BATCH_SIZE': 128,

          #minibatch中前景样本所占的比例
           'FG_FRACTION': 0.25,
          #与前景的overlap大于等于0.5认为该ROI为前景样本
           'FG_THRESH': 0.5,
          #与前景的overlap在0.0-0.5认为该ROI为背景样本
           'BG_THRESH_HI': 0.5,
           'BG_THRESH_LO': 0.0,

          # 训练bb回归器
           'BBOX_REG': True,
          #BBOX阈值，只有ROI与gt的重叠度大于阈值，这样的ROI才能用作bb回归的训练样本
           'BBOX_THRESH': 0.5,
          #归一化目标BBOX_NORMALIZE_TARGETS，减去经验均值，除以标准差
           'BBOX_NORMALIZE_TARGETS': True,
          #在BBOX_NORMALIZE_TARGETS为True时，归一化targets,使用经验均值和方差
           'BBOX_NORMALIZE_TARGETS_PRECOMPUTED': True,
          #Deprecated (inside weights) 弃用
           'BBOX_INSIDE_WEIGHTS': [1.0, 1.0, 1.0, 1.0],
           'BBOX_NORMALIZE_MEANS': [0.0, 0.0, 0.0, 0.0],
           'BBOX_NORMALIZE_STDS': [0.1, 0.1, 0.2, 0.2],

          #水平翻转图像，增加数据量
           'USE_FLIPPED': True, 
          #minibatch的两个图片应该有相似的宽高比，以避免冗余的zero-padding计算
            'ASPECT_GROUPING': True,
          #为产生的snapshot文件名称添加一个可选的infix.solver.prototxt指定了snapshot名称的前缀
           'SNAPSHOT_INFIX': '',
          #每迭代1000次产生一次snapshot
           'SNAPSHOT_ITERS': 10000,
          #在roi_data_layer.layer使用预取线程，作者认为不太有效，因此设为False
           'USE_PREFETCH': False,   
          -----------------rpn---------------
          #使用RPN检测目标
           'HAS_RPN': True,
          #batch size大小
           'RPN_BATCHSIZE': 256,

          #RPN的正样本阈值
           'RPN_POSITIVE_OVERLAP': 0.7,
          #RPN的负样本阈值
           'RPN_NEGATIVE_OVERLAP': 0.3,
          #如果一个anchor同时满足正负样本条件，设为负样本（应该用不到）
           'RPN_CLOBBER_POSITIVES': False,
          #前景样本的比例
           'RPN_FG_FRACTION': 0.5,
          #Deprecated (outside weights)弃用
           'RPN_BBOX_INSIDE_WEIGHTS': [1.0, 1.0, 1.0, 1.0],
          # proposal的高和宽都应该大于RPN_MIN_SIZE，否则，映射到conv5上不足一个像素点
           'RPN_MIN_SIZE': 16,
          
          #非极大值抑制的阈值
           'RPN_NMS_THRESH': 0.7,
          #在对RPN proposal使用NMS后，要保留的top scores的box数量
           'RPN_POST_NMS_TOP_N': 2000,
          #在对RPN proposal使用NMS前，要保留的top scores的box数量
           'RPN_PRE_NMS_TOP_N': 12000,
          #给定正RPN样本的权重 这里正负样本使用相同权重
           'RPN_POSITIVE_WEIGHT': -1.0
           },
  #使用GPU实施非极大值抑制
 'USE_GPU_NMS': True}
----------test--------------
 'TEST': {
          'SCALES': [600],
          'MAX_SIZE': 1000,
        #使用RPN生成proposal
          'HAS_RPN': True,
        #测试时非极大值抑制的阈值
          'NMS': 0.3,
        # 分类不再用SVM，设置为False
          'SVM': False,
        # 使用bb回归
          'BBOX_REG': True,
        # 使用selective_search生成proposal
          'PROPOSAL_METHOD': 'selective_search',
      -------rpn--------
          'RPN_MIN_SIZE': 16,
        #  RPN proposal的NMS阈值
          'RPN_NMS_THRESH': 0.7,
          'RPN_POST_NMS_TOP_N': 300,
          'RPN_PRE_NMS_TOP_N': 6000
        
        },
```

## demo.py
```
    #调用py-faster-rcnn/lib/fast_rcnn/test.py中的im_detect
    ##im的shape为(375,500,3)
    scores, boxes = im_detect(net, im) 
```



## test.py
```
    #转化图像大小为固定大小,改变后shape为（600，800)
    blobs, im_scales = _get_blobs(im, boxes) ##使用resize进行缩放到统一大小
    #im_info里记录了改变后的大小，及缩放比例，[ 600.,800.,1.60000002]
    blobs['im_info'] = np.array(
            [[im_blob.shape[2], im_blob.shape[3], im_scales[0]]],
            dtype=np.float32)
```


## 

```
{'queue': <multiprocessing.queues.Queue object at 0x4b97210>, 
'max_iters': 80000, 
'init_model': 'data/imagenet_models/ZF.v2.caffemodel', 
'solver': '/home/chengrenqing/py-faster-rcnn/models/pascal_voc/VGG16/faster_rcnn_alt_opt/stage1_rpn_solver60k80k.pt',
 'cfg': {
    'USE_GPU_NMS': True, 
    'ROOT_DIR': '/home/chengrenqing/py-faster-rcnn', 
    'GPU_ID': 0, 'EPS': 1e-14, 'RNG_SEED': 3,
     'MODELS_DIR': '/home/chengrenqing/py-faster-rcnn/models/pascal_voc', 'TRAIN': 
     {
        'RPN_BBOX_INSIDE_WEIGHTS': [1.0, 1.0, 1.0, 1.0], 
        'PROPOSAL_METHOD': 'selective_search', 
        'SNAPSHOT_ITERS': 10000, 
        'RPN_POST_NMS_TOP_N': 2000,
        'RPN_PRE_NMS_TOP_N': 12000, 
        'BBOX_REG': True, 'IMS_PER_BATCH': 2, 'RPN_FG_FRACTION': 0.5, 'BBOX_NORMALIZE_STDS': [0.1, 0.1, 0.2, 0.2], 'BATCH_SIZE': 128, 'ASPECT_GROUPING': True, 'USE_PREFETCH': False, 'BG_THRESH_LO': 0.0, 'FG_THRESH': 0.5, 'MAX_SIZE': 1000, 'BBOX_THRESH': 0.5, 
        'BG_THRESH_HI': 0.5, 'SCALES': [600], 'BBOX_NORMALIZE_TARGETS': True, 'BBOX_NORMALIZE_TARGETS_PRECOMPUTED': False, 'FG_FRACTION': 0.25, 'BBOX_NORMALIZE_MEANS': [0.0, 0.0, 0.0, 0.0], 'RPN_BATCHSIZE': 256, 'SNAPSHOT_INFIX': 'stage1', 'RPN_POSITIVE_WEIGHT': -1.0, 'RPN_NMS_THRESH': 0.7, 'USE_FLIPPED': True,
        'BBOX_INSIDE_WEIGHTS': [1.0, 1.0, 1.0, 1.0],
        'RPN_CLOBBER_POSITIVES': False, 'RPN_NEGATIVE_OVERLAP': 0.3,
        'HAS_RPN': False, 'RPN_MIN_SIZE': 16, 'RPN_POSITIVE_OVERLAP': 0.7
        },
         'MATLAB': 'matlab', 
         'TEST': 
         {
            'PROPOSAL_METHOD': 'selective_search', 'SVM': False, 'NMS': 0.3, 'RPN_NMS_THRESH': 0.7, 'SCALES': [600], 'RPN_POST_NMS_TOP_N': 300, 'HAS_RPN': True, 'RPN_PRE_NMS_TOP_N': 6000, 'BBOX_REG': True, 'RPN_MIN_SIZE': 16, 'MAX_SIZE': 1000
            }, 
            'DATA_DIR': '/home/chengrenqing/py-faster-rcnn/data', 'EXP_DIR': 'faster_rcnn_alt_opt', 'PIXEL_MEANS': array([[[ 102.9801,  115.9465,  122.7717]]]), 'DEDUP_BOXES': 0.0625}, 'imdb_name': 'voc_2007_trainval'}
```