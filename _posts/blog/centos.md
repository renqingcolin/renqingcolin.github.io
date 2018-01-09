---
layout: post
title: centos
categories: Blog
description: Faster rcnn 
keywords: Faster rcnn
---
##安装pip
```
yum -y install epel-release
yum install python-pip
pip install --upgrade pip
```
##Cuda 8.0
1. 下载[cuda8](https://developer.nvidia.com/cuda-80-ga2-download-archive)（与操作系统版本对应)  
2. sudo rpm -i cuda-repo-rhel7-8.0.61-1.x86_64.rpm
3. sudo yum clean al
4. sudo yum install cuda

## opencv

###依赖
yum install atlas-devel snappy-devel boost-devel leveldb leveldb-devel hdf5 hdf5-devel  glog glog-devel gflags gflags-devel protobuf protobuf-devel opencv opencv-devel lmdb lmdb-devel
```
sudo yum install python-devel
sudo pip install Cython
sudo pip install easydict
sudo pip install numpy

```


