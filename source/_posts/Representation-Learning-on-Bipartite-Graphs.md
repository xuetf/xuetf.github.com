---
title: Representation Learning on Bipartite Graphs
date: 2020-02-13 12:13:53
tags: [Paper,Graph Embedding,GCN,推荐系统,深度学习,Representation Learning,二分图]
comments: true
categories: 推荐系统
top: 17
---

本篇文章主要介绍图表示学习在推荐系统中常见的user-item interaction数据上的建模。我们可以将user-item interaction data看做是关于user-item的二分图Bipartite，这样可以基于GNN的方法进行user和item embedding的学习。主要介绍两篇文章，如下：

- KDD2018，GCMC：[Graph Convolutional Matrix Completion](https://www.kdd.org/kdd2018/files/deep-learning-day/DLDay18_paper_32.pdf)，在movie-len上的state of the art模型。
- SIGIR2019，NGCF：[Neural Graph Collaborative Filtering](https://arxiv.org/abs/1905.08108)，首次提出图协同过滤的概念。

<!--more-->

## GCMC

[KDD2018: Graph Convolutional Matrix Completion](https://www.kdd.org/kdd2018/files/deep-learning-day/DLDay18_paper_32.pdf)

### Introduction

这是一篇发表在KDD2018（二作T.N.Kipf是著名的 [GCN](https://openreview.net/pdf?id=SJU4ayYgl) 的一作）的文章。将矩阵补全问题看做是关于user-item 二分图的链接预测问题（a bipartite user-item graph with labeled edges），每种链接边可以看做是一种label（例如多种交互行为：点击，收藏，喜欢，下载等；在评分中，1~5分可以分别看做一种label）。有点像multi-view bipartite graph。作者提出了一个差异化信息传递(differentiable message passing)的概念，通过在bipartite interaction graph上进行差异化信息传递来学习结点的嵌入，再通过一个bilinear decoder进行链接预测。

### Formulation

给定关于用户-物品交互数据的二分图，$G=(\mathcal{W}, \mathcal{E}, \mathcal{R})$，其中$\mathcal{W}$是结点，包括用户结点$u_i \in \mathcal{W}_u, i=\\{1,...,N_u\\}$和物品结点$v_j \in \mathcal{W}_v, j=\\{1,...,N_v\\}$，$\mathcal{W}=\mathcal{W}_u \cup \mathcal{W}_v$. 边$(u_i ,r,v_j ) \in \mathcal{E}$代表了用户$u_i$和$v_j$的交互行为类型为$r$，$r \in \mathcal{R}=\\{1,...,R\\}$.其中$R$是总的交互类型种数。这里的交互行为原文中使用的是评分level来代表，如1~5分，则代表5种rating level，即：$R=5$。 

![gcmc](/picture/machine-learning/gcmc-demo.png)

可以看做是一个graph auto-encoders。Graph Encoder Model形如：$[Z_u ,Z_v ] = f (X_u ,X_v , M_1 , . . . , M_R)$。其中，$X_u, X_v$是用户或物品的Feature矩阵。$M_r \in \{0, 1\}^{N_u \times N_v}$是和交互类型$r$相关的邻接矩阵。$Z_u, Z_v$是输出的结点嵌入。Decoder形如：$\hat{M} = (Z_u ,Z_v)$用于预测$u$和$v$的链接类型或评分矩阵的评分值。然后使用重构误差最小化来训练模型，如RMSE或交叉熵。

### Models

#### Graph convolutional encoder

作者首先提出了一种针对不同rating level的sub-graph分别进行局部图卷积，再进行汇聚的encoder模型。这种局部图卷积可以看做是一种信息传递(message passing)，即：向量化形式的message在图中不同的边上进行传递和转换。每种边特定的信息传递过程如下(edge-type specific message)，以item $j$ 传递到 user $i$为例：
$$
\mu_{j \rightarrow i, r}= \frac{1}{c_{ij}} W_r x_j^v \tag{1}
$$
其中，$c_{ij}$是归一化常数因子，如$|\mathcal{N}(u_i)|$或$\sqrt{|\mathcal{N}(u_i) \mathcal{N}(v_j)}$。$W_r$是 edge-type specific parameter matrix。$x_j^v$是item $j$初始的特征，作者采用的是unique one-hot vector作为特征。相当于把$x_j^v$这一信息传递到了用户结点$i$上。这样针对每种类型的边，可以把所有$i$的邻域节点传递过来的信息做个累加操作，$\sum_{j \in \mathcal{N}_r(u_i)}{\mu_{j \rightarrow i, r}}$， $\text{for } r = 1,2,...,R$.

接着需要将不同edge-type采集到的信息进行一个汇聚，
$$
h_i^v = \sigma \left[\text{accum}\left(\sum_{j \in \mathcal{N}_1(u_i)}{\mu_{j \rightarrow i, 1}},...,\sum_{j \in \mathcal{N}_r(u_i)}{\mu_{j \rightarrow i , r}},..., \sum_{j \in \mathcal{N}_R(u_i)}{\mu_{j \rightarrow i, R}}\right)\right] \tag{2}
$$
上式的$\text{accum}$是一个汇聚操作，例如可以采用连接操作或求和操作，$\sigma$ 是激活函数，例如Relu。 完整的式(2)称为

graph convolution layer，可以叠加多个graph convolution layer。

为了得到用户$u_i$最终的embedding，作者加了个全连接层做了个变换：
$$
z_{i}^u = \sigma(W h_i^u) \tag{3}
$$
物品$v_i$的embedding和用户$u_i$的求解是对称的，只不过使用的是两套参数。

#### Bilinear decoder

为了重构二分图上的链接，作者采用了bilinear decoder，把不同的rating level分别看做一个类别。bilinear operation后跟一个softmax输出每种类别的概率：
$$
p(\hat{M}_{ij}=r)=\text{softmax}((z_i^u)^T Q_r z_j^v) \tag{4}
$$
$Q_r$是可训练的edge-type r特定的参数。

则最终预测的评分是关于上述概率分布的期望：
$$
\hat{M_{ij}} = g(u_i ,v_j) = E_{p(\hat{M}_{ij}=r)}[r] = r p(\hat{M}_{ij}= r) \tag{5}
$$

#### Model Training

最小化 negative log likelihood of the predicted ratings $\hat{M}_{ij}$.
$$
\mathcal{L} = − \sum_{i,j;Ω_{ij}=1} \sum_{r=1}^R I[M_{ij}=r] \log p(\hat{M}_{ij}=r) \tag{6}
$$
作者在训练过程中，还使用了一些特殊的优化点。

- **Node dropout**

  对特定的某个节点，随机地将outgoing的信息随机drop掉。这能够有效的提高模型的鲁棒性。例如：去除掉某个结点时，模型性能不会有太大的变化。

- **Weight sharing**

  不是所有的用户或物品在不同rating level上都拥有相等的评分数量。这会导致在局部卷积时，$W_r$上某些列可能会比其他列更少地被优化。因此需要使用某种在不同的$r$之间进行参数共享的方法。
  $$
  W_r = \sum_{s=1}^r T_s \tag{7}
  $$
  $T_s$是基础矩阵。也就是说越高评分，$W_r$包含的$T_s$数量越多。



  作者采用了一种基于基础参数矩阵的线性组合的参数共享方法：
$$
  Q_r = \sum_{s=1}^{n_b} a_{rs} P_s \tag{8}
$$
  其中，$n_b$是基础权重矩阵的个数，$P_s$是基础权重矩阵。$a_{rs}$是可学习的系数。

#### Side Information

为了建模结点的辅助信息，作者在对$h_i^u$做全连接层变换时，考虑了结点的辅助信息:
$$
z_{i}^u = \sigma(W h^u_i +W_2^{u,f} f_i^u), f_i^u = σ(W_1^{u,f} x_i^{u,f} + b^u) \tag{9}
$$
即：先对特征$x_i^{u,f}$做一个变换得到$f_i^u$。与$h^u_i$分别经过线性变换后加起来并激活，最终得到结点的嵌入表示。

#### Summary

总结一下整体的过程，模型的核心过程形式化如下：
$$
\boldsymbol{z}_i^u, \boldsymbol{z}_j^v = \text{GNN-Encoder}(\mathcal{G}_{u,i}) \tag{10}
$$

$$
p(\hat{M}_{ij}=r)= \text{softmax}\left(\text{Bilinear-Decoder}(\boldsymbol{z}_i^u, \boldsymbol{z}_j^v)\right) \tag{11}
$$

$$
\mathcal{L} = − \sum_{i,j;Ω_{ij}=1} \sum_{r=1}^R I[M_{ij}=r] \log p(\hat{M}_{ij}=r) \tag{12}
$$

$$
\hat{M_{ij}} = g(u_i ,v_j) = E_{p(\hat{M}_{ij}=r)}[r] = r p(\hat{M}_{ij}= r) \tag{13}
$$

先经过GNN-Encoder输出user和item的嵌入表示（公式10），再经过Bilinear Decoder输出属于不同评分值的概率（公式11），最后公式(12)是关于评分多分类问题的交叉熵损失。具体预测的时候使用公式(13)预测评分，进行topk推荐。

## NGCF

[SIGIR2019: Neural Graph Collaborative Filtering](https://arxiv.org/abs/1905.08108)

### Intuition

这是何向南教授团队在SIGIR2019发表的一篇文章，主要针对隐式反馈行为数据。为了解决传统的方法主要通过user或item的pre-existing fetures的映射得到user和item的embedding，而缺乏了user和item之间重要的协同信息(collaborative signal)这一问题，作者提出了一种新的推荐框架——神经图协同过滤。这篇文章核心的目标问题在于如何更好的将user-item之间的协同信息建模到user和item的emebdding表示中。
```
We develop a new recommendation framework Neural Graph Collaborative Filtering (NGCF), 
which exploits the user-item graph structure by propagating embeddings on it. 
This leads to the expressive modeling of high-order connectivity in user-item graph, 
effectively injecting the collaborative signal into the embedding process in an explicit manner.
```


传统的方法无法学到很好的Embedding，归咎于缺乏对于重要的**协同信息**显示地进行编码过程，这个协同信息隐藏在user-item交互数据中，蕴含着users或者items之间的行为相似性。更具体地来说，传统的方法主要是对ID或者属性进行编码得到embedding，再基于user-item interaction定义重构损失函数并进行解码。可以看出，user-item interaction只用到了解码端，而没有用到编码端。这样会导致，学习到的embedding的信息量不够，为了更好的预测，只能依赖于复杂的interaction (decoder) function来弥补次优的embedding在预测上的不足。
```
The key reason is that the embedding function lacks an explicit encoding of the crucial collaborative signal, 
which is latent in user-item interactions to reveal the behavioral similarity between users (or items)

To be more specific, most existing methods build the embedding function with the descriptive features only (e.g., ID and attributes), 
without considering the user-item interactions which are only used to define the objective function for model training. 
As a result, when the embeddings are insufficient in capturing CF, 
the methods have to rely on the interaction function to make up for the deficiency of suboptimal embeddings.
```

难就难在如何将interaction中的collaboritive signals建模到embedding的表示中。由于原始的interaction数据规模通常比较庞大，使得难以发掘出期望的collaboritive signals（distill the desired collaborative signal）。

作者在这篇文章中提出基于user-item intractions中的high-order connectivity来解决collaboritive signals的发掘和embedding的建模，做法是在user-item interaction graph structure图结构中进行collaborative signal的编码。
```
we tackle the challenge by exploiting the high-order connectivity from useritem interactions, 
a natural way that encodes collaborative signal in the interaction graph structure.
```


关于high-order connectivity含义的可视化如下，$u_1$的两种视角：二分图 和 树

![high-order](/picture/machine-learning/high-order.png)

### Keys

作者设计了一个神经网络来迭代地在二部图上进行embeddings的传播，这可以看做是在embedding空间构建信息流。
```
We design a neural network method to propagate embeddings recursively on the graph, 
which can be seen as constructing information flows in the embedding space.
```


特别地，作者设计了一种**embedding传播层**（embedding propagation layer），通过汇聚交互过的items或users来提炼出users或items的embeddings。进一步，通过叠加多个embedding propagation layer，可以使得embeddings来捕获二部图中high-order connectivities所蕴含的协同信息。
```
Specifically, we devise an embedding propagation layer, which refines a user’s (or an item’s) embedding 
by aggregating the embeddings of the interacted items (or users). By stacking multiple embedding propagation layers, 
we can enforce the embeddings to capture the collaborative signal in high-order connectivities.
```



### Models

模型的结构如下：

![ngcf](/picture/machine-learning/NGCF.png)

#### Embedding Layer

嵌入层，offers and initialization of user embeddings and item embeddings。
  $$
  \boldsymbol{E}=[\boldsymbol{e}_{u1},...,\boldsymbol{e}_{uN} \ , \ \boldsymbol{e}_{i1},...,\boldsymbol{e}_{iM}] \tag{1}
  $$
传统的方法直接将$E$输入到交互层，进而输出预测值。而NGCF将$E$输入到多层嵌入传播层，通过在二部图上进行嵌入的传播来对嵌入进行精炼。

#### Embedding Propagation Layer

嵌入传播层，refine the embeddings by injecting high-order connectivity relations。

包括两个步骤，Message construction和Message aggregation。这个部分和KDD2018的GCMC那篇文章是类似的描述方式。

##### Message Construction

对于每个user-item pair $(u,i)$, 定义从$i$到$u$传递的message如下：
    $$
    \boldsymbol{m}_{u \leftarrow i}=f(\boldsymbol{e_i}, \boldsymbol{e_u}, p_{ui}) \tag{2}
    $$
    其中，$\boldsymbol{m}_{u \leftarrow i}$定义为message embedding。$f(\cdot)$是message encoding function，将user和item的 embedding $\boldsymbol{e}_u, \boldsymbol{e}_i$作为输入，并使用$p_{ui}$来控制每条边edge $(u,i)$在每次传播中的衰减系数。

具体的，作者使用如下message encoding function:
    $$
    \boldsymbol{m}_{u \leftarrow i} = \frac{1}{\sqrt{|\mathcal{N}_u||\mathcal{N}_i|}}(\boldsymbol{W}_1 \boldsymbol{e}_i + \boldsymbol{W}_2(\boldsymbol{e}_i \odot \boldsymbol{e}_u)) \tag{3}
    $$
    可以看出作者不仅考虑了message的来源$\boldsymbol{e}_i$（传统图卷积方法只考虑这个），还考虑了信息来源和信息目的地之间的关系，即$\boldsymbol{e}_i \odot \boldsymbol{e}_u$，这个element-wise product是典型的特征交互的一种方式，值得学习。$p _u=\frac{1}{\sqrt{|\mathcal{N}_u||\mathcal{N}_i|}}$，$\mathcal{N}_u$是用户$u$的1-hop neighbors。从表示学习角度：$p_{ui}$反映了历史交互item对用户偏好的贡献程度。从信息传递角度，$p_{ui}$可以看做是折扣系数，随着传播路径长度的增大，信息慢慢衰减（这个可以通过叠加多层，并代入到式子，会发现前面系数也是连乘在一起了，说明路径越长，衰减越大）。

##### Message Aggregation

这个阶段，我们将从用户$u$的邻居传递过来的信息进行汇聚来提炼用户$u$的嵌入表示。
    $$
    \boldsymbol{e}_u^{(1)} = \text{LeakyReLU}\left(\boldsymbol{m}_{u \leftarrow u} + \sum_{i \in \mathcal{N}_u} \boldsymbol{m}_{u \leftarrow i} \right) \tag{4}
    $$
    $\boldsymbol{e}_u^{(1)}$是经过一层嵌入传播层得到的提炼后的用户$u$嵌入表示。LeakyReLU允许对于正反馈信息和少部分负反馈信息的编码。$\boldsymbol{m}_{u \leftarrow u} = \boldsymbol{W}_1 \boldsymbol{e}_u$则考虑了self-connection，$\boldsymbol{W}_1$和Equation(2)中的$\boldsymbol{W}_1 $是共享的。

  item的嵌入表示同理可得。embedding propagation layer的好处在于显示地挖掘 first-order connectivity信息来联系user和item的表示。 

  得到的$\boldsymbol{e}_u^{(1)}$可以作为下一个embedding propagation layer的输入，通过叠加多层，可以挖掘multi-hop的关联关系。迭代式如下：
  $$
  \boldsymbol{e}_u^{(l)} = \text{LeakyReLU}\left(\boldsymbol{m}^{(l)}_{u \leftarrow u} + \sum_{i \in \mathcal{N}_u} \boldsymbol{m}^{(l)}_{u \leftarrow i} \right) \tag{5}
  $$
  其中，
  $$
  \boldsymbol{m}^{(l)}_{u \leftarrow i} = p_{u,i}(\boldsymbol{W}^{(l)}_1 \boldsymbol{e}^{(l-1)}_i + \boldsymbol{W}_2^{(l)}(\boldsymbol{e}_i^{(l-1)} \odot \boldsymbol{e}_u^{(l-1)})) \tag{6}
  $$

  $$
  \boldsymbol{m}^{(l)}_{u \leftarrow u} = \boldsymbol{W}^{l}_1 \boldsymbol{e}_u^{(l-1)}
  $$

  进一步，作者给出了上述Equation(5), (6) 过程的矩阵表示形式，有助于实现layer-wise propagation。
  $$
  \underbrace{\boldsymbol{E}^{(l)}}_{\mathbb{R}^{(N+M) \times d_l}} = \text{LeakyReLU} \left( \underbrace{(\boldsymbol{\mathcal{L}} + \boldsymbol{I})}_{\mathbb{R}^{(N+M)\times (N+M)}} \overbrace{\boldsymbol{E}^{(l−1)}}^{\mathbb{R}^{(N+M) \times d_{l-1}}} \underbrace{\boldsymbol{W}_1^{(l)}}_{\mathbb{R}^{d_{l-1} \times d_l}} + \boldsymbol{\mathcal{L}} \underbrace{\boldsymbol{E}^{(l−1)}}_{(N+M) \times d_{l-1}} \odot \overbrace{\boldsymbol{E}^{(l−1)}}^{(N+M) \times d_{l-1}} \underbrace{\boldsymbol{W}_2^{(l)}}_{\mathbb{R}^{d_{l-1} \times d_l}} \right) \tag{7}
  $$

  其中，$E^{(l)} \in \mathbb{R}^{(N +M)×d_l}$，即：把user, item的embeddings矩阵concat在一起，一起进行传播。也就是说，上述是user，item共同进行传播的表示形式，因此所有的矩阵都是concat在一起的形式。

  作者说，$\boldsymbol{\mathcal{L}}$表示user-item interaction graph的拉普拉斯矩阵，$\boldsymbol{\mathcal{L}}=\boldsymbol{D}^{-1/2} \boldsymbol{A} \boldsymbol{D}^{-1/2}$，其中，$\boldsymbol{A} \in \mathbb{R}^{(N+M) \times (N+M)}$是邻接矩阵，是user-item 交互矩阵和item-user交互矩阵构成的，即：$\boldsymbol{A} = \left[ \begin{array}{ccc}
   \boldsymbol{0} & \boldsymbol{R} \\
   \boldsymbol{R}^T & \boldsymbol{0} \\
  \end{array} \right]  $。(但是我个人记得拉普拉斯矩阵三种方式都不是长这个样子的，有可能这些定义之间差异很小，可能特征根是一样的，故也叫拉普拉斯矩阵吧，虽然和标准的拉普拉斯矩阵之间有一丝差异。)

  这个矩阵表示形式很好理解，主要点在于，和$\boldsymbol{\mathcal{L}}$的**乘法**就对应于公式(5)中$\sum_{i \in \mathcal{N}_u} \boldsymbol{m}^{(l)}_{u \leftarrow i}$对邻域节点的汇聚**求和操作**，其它的一目了然。

#### Prediction Layer

预测层，aggregates the refined embeddings from different propagation layers and outputs the affinity score of a user-item pair.

  最终的嵌入表示是原始的embedding和所有嵌入传播层得到的embedding全部concat在一起的结果。即：
  $$
  \boldsymbol{e}_u^{\*}= \text{concat}(\boldsymbol{e}_u^{(0)},\boldsymbol{e}_u^{(1)},...,\boldsymbol{e}_u^{(L)}) \tag{8}
  $$
  $\boldsymbol{e}_u^{(0)}$是初始化的embeddings。$\boldsymbol{e}_i^{*}$同理可得。

  最后预测user-item交互的时候使用点乘：
  $$
  \hat{y}_{\text{NGCF}}(u,i) = {\boldsymbol{e}_u^{\*}}^T \boldsymbol{e}_i^{\*}
  $$

#### Loss

最后作者采用的是pairwise BPR loss进行优化。
$$
Loss = − \sum_{(u,i,j) \in O}\ln \sigma(\hat{y}_{ui} − \hat{y}_{uj}) + \lambda||\Theta||_2^2
$$
其中，$O = \\{(u,i, j)|(u,i) \in R^+ , (u, j)  \in R^- \\}$， $R^+$是观测数据，$R^-$是未观测数据，$\Theta=  \\{ \boldsymbol{E}, \\{ {\boldsymbol{W}_1^{(l)} , \boldsymbol{W}_2^{(l)}\\}}_{l=1}^L \\}$是所有的可训练参数。

## Summary

关于图二分图的挖掘，是推荐系统中的核心问题之一。传统的图表示学习直接对二部图进行挖掘是有问题的。个人认为传统的graph embedding在建模时实际上很依赖于graph的手动构造，连边权重的合理设置，且没有把二分图中的collaborative signals直接建模到结点的embedding表示中，而只是通过目标函数来间接引导。而二分图的构建是基于原始数据的，没有连边权重，且数据过于稀疏，蕴含在二分图中的协同信息是非常隐晦的和难以挖掘的，故传统的图表示学习方法很难处理。本篇文章中介绍的两个工作基于message passing范式的GNN来挖掘二分图，将collaborative signals直接建模到embedding中，这是非常值得借鉴的。另外，这两篇文章的github都有源码，且代码风格都不错，值得学习和实践。

## References

[KDD2018，GCMC：Graph Convolutional Matrix Completion](https://www.kdd.org/kdd2018/files/deep-learning-day/DLDay18_paper_32.pdf)
[SIGIR2019，NGCF：Neural Graph Collaborative Filtering](https://arxiv.org/abs/1905.08108)
[Github, GCMC, Source Code](https://github.com/riannevdberg/gc-mc)
[Github, NGCF, Source Code](https://github.com/xiangwang1223/neural_graph_collaborative_filtering)
