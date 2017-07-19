---
layout: post
title: TensorFlow学习笔记
categories: Blog
description: TensorFlow
keywords: TensorFlow
---
# TensorFlow学习笔记1--基本概念

## 计算图

TensorFlow中的每一个计算都是计算图中的一个节点，而节点之间的边描述了计算之间的依赖关系。在TensorFlow中，系统会自动维护一个默认的计算图，通过`tf.get_default_graph`函数可以获取当前的计算图，除了使用默认的计算图，TensorFlow支持通过`tf.Graph`函数来生成新的计算图。不同计算图上的张量和运算都不会共享。

 小结：计算图是TensorFlow的计算模型，所有TensorFlow的程序都会通过计算图的形式表示。计算图上的每一个节点都是一个运算，而且计算图上的边则表示运算之间的数据传递关系。计算图上还保存了运行每个运算的设备信息（比如是通过CPU上还是GPU运行）以及运算之间的依赖关系。计算图提供了管理不同集合的功能，并且TensorFlow会自动维护五个不同的默认集合。

## 张量
 
#### 概念

 所有的数据都是通过张量的形式表示。从功能的角度上看，张量可以被简单理解为多维数组。当张量在TensorFlow中的实现并不是直接采用数组的形式，它只是对TensorFlow中运算结果的引用。在张量中并没有真正保存数字，它保存的是如何得到这些数字的计算过程。
 一个张量中主要保存了三个属性：名字(name)、维度(shape)和类型(type)
 >名字：不仅是一个张量的唯一标识符，它同样也给出了这个张量是如何计算出来的。
 >
 >维度：这个属性描述了一个张量的维度信息
 >
 >类型：每一个张量会有一个唯一的类型。TensorFlow会对参与运算的所有张量进行类型检测，当发现类型不匹配时会报错。

#### 使用

第一类用途是对中间计算结果的引用。当一个计算包含很多中间结果时，使用张量可以大大提高代码的可读性。
使用张量的第二类情况是当计算图构造完成后，张量可以用来获得计算结果，也就是得到真实的数字

小结：张量是TensorFlow的数据模型，TensorFlow中所有运算的输入、输出都是张量。张量本身并不存储任何数据，它只是对运算结果的引用。通过张量，可以更好的组织TensorFlow程序。

## 会话

TensorFlow中的会话执行定义好的运算，会话拥有并管理TensorFlow程序运行时的所有资源。当所有的计算完成之后需要关闭会话来帮助系统回收资源，否则就可能出现资源泄露的问题。
TensorFlow中使用会话的模式一般有两种，第一种模式需要明确调用会话生成函数和关闭会话函数，这种模式的代码流程如下：

```
    #创建一个会话。
    sess = tf.Session()
    #使用这个创建好的会话来得到关系的运算的结果。比如
    sess.run(...)
    #关闭会话使得本次运行中使用到的资源可以被释放。
    sess.close()
```
这种模式有个问题，就是当程序因为异常而退出时，关闭会话的函数可能就不会被执行从而导致资源泄露。为了解决异常退出时资源释放的问题，TensorFlow可以通过Python的上下文管理器来使用会话。如：
```
    #创建一个会话，并通过Python中的上下文管理器来管理这个会话
    with tf.Session() as sess:
        #使用这个创建好的会话来计算关心的结果。
        sess.run(...)
    #不需要再调用“Session.slose()”函数来关闭会话
    #当上下文退出时会话关闭和资源释放也自动完成了。
```
小结：会话是TensorFlow的运算模型，它管理了一个TensorFlow程序拥有的系统资源，所有运算都要通过会话执行。

# TensorFlow学习笔记2--TensorFlow实现神经网络

## 前向传播算法

这里讲的是全连接网络结构的前向传播算法。前向传播就是通过上一层的结点以及对应的连接权值进行加权和运算，最终结果再加上一个偏置项，最后在通过一个非线性函数（即激活函数），如ReLu，sigmoid等函数，最后得到的结果就是本层结点w的输出。 
最终不断的通过这种方法一层层的运算，得到输出层结果。 
以下样例介绍了如何通过变量实现神经网络的参数并进行向前传播
```
    import tensorflow as tf
    #声明w1、w2两个变量。这里还通过seed参数设定了随机种子。
    #这样可以保证每次运行得到的结果是一样的
    #TensorFlow中的变量的初始值可以设置成随机数、常数或者是通过其他变量的初始值计算得到。
    w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
    w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1)

    #展示将输入的特征向量定义为一个常量。追这里x是一个1*2的矩阵
    x = tf.constant([0.7,0.9])

    #通过前向传播算法获得神经网络输出
    a = tf.matmul(x,w1)
    y = tf.matmul(a,w2)

    sess = tf.Session()
    #这里不能直接通过sess.run(y)来获取y的取值
    #因为w1和w2都还没有运行初始化过程。下面的两行分别初始化了w1和w2两个变量
    sess.run(w1.initializer) #初始化w1
    sess.run(w2.initializer) #初始化w2
    #输出结果
    print(sess.run(y))
    sess.close()
```

注：
1、上面使用initializer来初始化变量，当变量数目增多，或者变量之间存在依赖关系时，单个调用的方案就比较麻烦了。为了解决这个问题，可以使用`tf.initialize_all_variables`函数来实现初始化所有变量的过程，这个函数也会自动处理变量之间的依赖关系。
```
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
```
2、一般所有的变量都会自动的加入GraphKeys.VARIABLES这个集合，通过`tf.all_variables`函数可以得到当前计算图上所有的变量。当构建机器学习模型时，比如神经网络，可以通过变量声明函数中的trainable参数来区分需要优化的参数（比如神经网络中的参数）和其他参数（比如迭代的轮数）。如果声明变量是参数trainable为True，那么这个变量将会被加入GraphKeys.TRAINABLE_VATIABLES集合。在TensorFlow中可以通过tf.trainable_variables函数得到所有需要优化的参数。

## 一个完成的程序训练数据网络解决二分类问题
```
#coding:utf-8
import tensorflow as tf 
from numpy.random import RandomState
#定义训练数据batch的大小
batch_size = 8

#定义神经网络的参数
w1 = tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 = tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

#在shape的一个维度上使用None可以方便使用不大的batch大小。在训练需要把数据扥城比较小的batch
#但在测试时，可以一次性使用全部的数据。当数据集比较小时这样比较方便测试，但数据集比较大时，将大量数据放在一个batch可能会导致内存溢出。
x = tf.placeholder(tf.float32,shape=(None,2),name='x-input')
y_ = tf.placeholder(tf.float32,shape=(None,1),name='y-input')

#定义神经网络前向传播过程。
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

#定义损失函数和反向传播的算法
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

#通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size,2)
#定义规则来给出样本的标签。在这里所有x1+x2<1的样例都被认为是正样本（比如零件合格），
#而其他为负样本（比如零件不合格）。和TensorFlow游乐场中的表示法不大一样的地方是，
#在这里使用0来表示负样本，1表示正样本。大部分解决分类问题的神经网络都会采用
#0和1的表示方法
Y = [[int(x1+x2<1)] for (x1,x2) in X]

#创建一个会话来运行TensorFlow程序。
with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    #初始化变量
    sess.run(init_op)
    print sess.run(w1)
    print sess.run(w2)

    #设置训练轮数
    STEPS = 5000
    for i in range(STEPS):
        #每次选取batch_size个样本进行训练
        start = (i * batch_size) % dataset_size
        end = min(start+batch_size,dataset_size)

        #通过选取的样本训练神经网络并更新参数。
        sess.run(train_step,feed_dict = {x:X[start:end],y_:Y[start:end]})
        if i % 1000 == 0:
            #每隔一段时间计算在所有数据上的交叉熵并输出
            total_cross_entropy = sess.run(cross_entropy,feed_dict = {x: X,y_: Y})
            print("After %d training step(s),cross entropy on all data is %g"%(i,total_cross_entropy))
    print sess.run(w1)
    print sess.run(w2)

```
从这段程序可以总结出训练神经网络的过程可以分为以下3个步骤：

    1. 定义神经网络的结构和前向传播的输出结构。
    2. 定义损失函数以及选择反向传播优化的算法
    3. 生成会话(`tf.Session`)并且在训练数据上反复运行反向传播优化算法。
无论神经网络的结构如何变化，这三个步骤是不变的。

# TensorFlow学习笔记3--深层神经网络

### 线性
#### 线性模型局限性
在线性模型中，模型的输出为输入的加权和，线性模型的最大特点是任意线性模型的组合仍然还是线性模型。在两层神经网络（不算输入层），其前向传播完全符合线性模型的定义，以此类推，只要通过线性变换，任意层的全连接神经网络和单层神经网络模型的表达能力没有任何区别，而且它们都是线性模型，然而线性模型能够解决的问题是有限的，这就是线性模型最大的局限性。
#### 激活函数实现去线性化
激活函数是通过一些非线性的函数来使得神经网络模型不再是线性的，主要有两个改变：第一个改变是增加了偏置项，偏置项是神经网络中非常常用的一种结构。第二个改变就是每个节点的取值不再是单纯的加权和，每个阶段的输出在加权和的基础上还做了一个非线性变换。
常用的激活函数：

| 函数名 | 表达式 | TensorFlow对应函数 |
| --- | --- | ---|
| ReLU  |$f(x)=max(x,0)$ | `tf.nn.relu` |
| sigmoid  |$f(x)=\frac 1 {1+e^{-x}}$ | `tf.sigmoid` |
| tanh  |$f(x)=\frac {1-e^{-2x}} {1+e^{-2x}}$ | `tf.tanh` |

### 损失函数
#### 分类
交叉熵是常用的评价一个输出和期望输出的向量的距离，刻画了两个概率分布之间的距离，它是分类问题中使用比较广的一种损失函数。给定两个概率分布$p$和$q$，通过$q$来表示$p$的交叉熵为：
\[H(p,q)=-\sum _x p(x)log q(x)\]
交叉熵刻画的是通过概率分布$q$来表达概率分布$p$的困难程度，因为正确答案是希望得到的结果，所以当交叉熵作为神经网络的损失函数时，$p$代表的是正确答案，$q$代表的是预测值。
注意交叉熵刻画的是两个概率分布之间的距离，然而神经网络的输出却不一定是一个概率分布。如何将神经网络前向传播得到的结果也变成一个概率分布呢？Softmax回归就是一个非常常用的方法。假设原始的神经网络输出为$y_1,y_2,\cdots,y_n$，那么经过Softmax回归处理之后的输出为：
\[softmax(y)_i = y_i^` = \frac {e^{y_i}} {\sum_{j=1}^n e^{y_j}} \]
从以上公式中可以看出，原始神经网络的输出被用置信度来生成新的输出，而新的输出满足概率分布的所有要求。这个新的输出可以理解为经过神经网络的推倒，一个样例为不同类别的概率分别是多大。这样就把神经网络的输出也变成了一个概率分布，从而可以通过交叉熵来计算预测的概率分布于真实答案的概率分布之间的距离了。
#### 回归
与分类问题不同，回归问题解决的是对具体数值的预测，需要预测的不是一个实现定义好的类别，而是一个任意实数。解决回归问题的神经网络一般只有一个输出节点，这个节点的输出值就是预测值。对于回归问题，最常用的损失函数是均方误差，它的定义如下：
\[MSE(y,y^、)=\frac {\sum_{i=1}^n (y_i-y_i^、)^2}{n}\]
