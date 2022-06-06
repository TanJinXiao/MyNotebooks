# Deep Residual Learning for Image Recognition

## 1. 文章想要解决的问题

较深的神经网络难以训练。也就是深度神经网络的退化问题。

CNN整合了低中高不同层次的特征，特征的层次可以通过加深网络来丰富。因此在构建卷积网络时，网络的深度越高，可抽取的特征层次也就越丰富。但是较深的网络不一定会带来更好的表现。例如：

<img src="../image/1_Deep Residual Learning for Image Recognition/1.1 Figure 1.png" style="zoom:67%;" />

该现象会由什么原因造成。

1. 首先排除过拟合，因为过拟合是让网络在训练集上表现的更好而测试集更差。它的表现是高方差低偏差，训练集误差小而测试集误差大。
2. 梯度消失/爆炸。梯度消失/爆炸是神经网络在反向传播的时候，反向连乘的梯度小于1（或者大于1），导致连乘次数变多以后（层次加深），传回首层的梯度过小甚至为0（过大或无穷大）。 这个问题是从一开始就阻碍收敛，目前很大程度上可以通过normalized initialization(标准初始化？)和intermediate normalization layers(中间的正则化层，BN层)规整数据分布来解决该问题。(SGD+反向传播)
3. 退化问题。随着网络深度的增加，精度会逐渐饱和，然后迅速退化。



本文提出了一个残差学习框架(residual learning framework)来简化训练。

## 2. 本文采用的新知识

提出了一个残差学习框架(residual learning framework)。如果添加的层可以构造为恒等映射(identity mapping)，那么较深的模型的训练误差应该不大于较浅的模型。

通过残差学习来重构模型，进行预处理，如果恒等映射是最优的，求解器可以简单的将多个非线性层的权值趋近于零来毕竟恒等映射。

### 2.1. 残差学习 

<img src="../image/1_Deep Residual Learning for Image Recognition/fig. 2.png" style="zoom:80%;" />

图为残差网络的一个"shortcut connection"(快捷连接)结构。网络不在直接拟合原先的映射，而是拟合残差映射。将所需的底层映射表示为$H(x)$，让原本叠加的非线性层拟合另一个映射$F(x)=H(x)-x$，则原始映射被重新定义为$F(x)+x$。假设它相比于原始的、未引用的映射更容易去优化残差映射。再极端情况下，如果恒等映射是最优的，它更容易推动残差为零而不是通过堆叠非线性层来拟合恒等映射。【意思是$F(x)$是残差？如果恒等映射$x=x$是最优的，那么相比于以前的推动$F(x)=x$，推动$F(x)=0$而使得$H(x)=x$更容易】

### 2.2. 

## 3. 本文贡献

通过在ImageNet数据集的综合实验，证明了1）当深度增加时，极深的残差网络很容易优化，而普通网络表现出较高的训练误差。2）深度残差网络可以轻松的在增加深度时获取更好的精度，产生的结果比以前的网络好的多。

### 

## 论文相关资源

### 相关论文

[1]. [ImageNet Classification with Deep Convolutional Neural Networks](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)

### 代码是否开源

### 数据集