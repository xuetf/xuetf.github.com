---
layout: false
title: Spectral
---
# Spectral Clustering

　　谱聚类（$spectral \ clustering$）是最广泛使用的聚类算法之一。通常分析数据前，我们想获取对数据的第一直观印象，初步识别数据中的某种相似性。本文将介绍谱聚类算法，比起传统的K-Means算法，谱聚类对数据分布的适应性更强，聚类效果也很优秀，同时聚类的计算量也小很多，更加难能可贵的是实现起来也不复杂。下面我们就对谱聚类的算法原理做一个总结。

## 图论基础

　　谱聚类基于图论，也就是我们聚类过程中，实际上会将所有样本点都看成图中的顶点，谱聚类的本质是将聚类问题转化为一个图上关于顶点划分的最优问题。

　　图G由顶点集和边集构成，记作$G=(V,E)$，这里考察加权无向图，每两个顶点之间连边的权重都是非负的，并使用图G的邻接矩阵$adjacency \ matrix$，$W=(w_{ij})_{i,j=1,...,n}$来表示。

　　顶点的度定义为：$d_i=\sum_{j=1}^n w_{ij}$, 该式子实际上只对和$v_i$相连的顶点之间的边进行求和，和$v_i$不相连的顶点$w_{ij}=0$，实际上$d_i$就是邻接矩阵行和。这里定义度矩阵$Degree \ matrix$， $D$为以$d_1, ..., d_n$为对角元素的对角矩阵。

　　对于给定的某个顶点集$A \subset V$, 定义$A$的补为$V \backslash A$ ，记作$\bar{A}$。并定义指示向量$indicator\  vector$, $\mathbf{1}_A=(f_1,f_2,...,f_n)^T$，该指示向量作用于所有顶点，若$f_i=1$, 代表$v_i \in A$, 否则$f_i=0$。为了方便，对于A集合的顶点，简记为$i \in A$。

　　对于两个不相交的顶点集$A,B$, 定义，
$$
W(A,B):=\sum_{i \in A ,j \in B} w_{ij}
$$
　　考察两种衡量顶点集A大小的方法：$|A|$考察顶点集A中的顶点个数，$vol(A)$将所有与顶点集A中的顶点相连的边的权重累加起来。
$$
|A|:=the \ number \ of \  vertices \ in \ A \\
vol(A) := \sum_{i \in A} d_i
$$
　　定义顶点集$A \subset V$构成的子图为连通的，当且仅当$A$中任意两个顶点都能通过一条路径连通，且路径上的所有中间顶点都在$A$上。

　　定义顶点集$A \subset V$是连通分量($connected \ component$), 当且仅当A构成的子图是连通的，且$A$和$\bar{A}$中的顶点不存在相连的边。也就是连通的只要求$A$中任意两个顶点$a,b$存在只经过A中顶点的路径，$a$可能存在与$\bar{A}$某个顶点相连的边，连通分量进一步约束$A$和$\bar{A}$的顶点不存在相连的边。

　　将$V$划分为若干连通分量，$A_1, A_2,...,A_k$，且$A_i \cap A_j=\emptyset，A_1 \cup A_2...\cup A_k=V$ 。

## 相似图

　　实际上给定样本集，若把样本看成图上的顶点，此时还构不成图，因为顶点和顶点之间不存在边相连。因此，我们需要设计边，且边上的权重需要使用某种度量方式进行表示，也即需要为邻接矩阵上的值进行刻画。

　　一种方式是定义边权重为边相连的两个顶点之间的相似性，任意两个顶点相连当且仅当这两个顶点的相似性$s_{ij}$为正数或大于某个阈值，此时边权重就用$s_{ij}$来代表。

　　因此，聚类问题可以转化成图切割问题，使得聚类族之间的边权重较低，换句话说，不同聚类族之间的顶点相似性低，这也就意味着这两个聚类族相似性较低。同时，使得聚类族内部的顶点相似性较高。

　　剩下的问题是如何度量样本之间的相似性？基本思想是，距离较远的两个点之间的相似性较低，距离近的点之间相似性高($model\  local\ neighborhood\ relationships\ between\ the\ data\ points$)。

　　有三种方法，$\epsilon-$邻近法、$k$近邻法、全连接法。

### $\epsilon$\-邻近法

　　$\epsilon-$邻近法，设置一个距离阈值$\epsilon$, 使用欧式距离度量样本间的相似性$s_{ij} = ||x_i-x_j||_2^2$, 再根据$s_{ij}， \epsilon$大小关系来确定边权重。
$$
w_{ij}= \begin{cases} 0& {s_{ij} > \epsilon} \\ \epsilon& {s_{ij} \leq \epsilon} \end{cases}
$$
　　从中可以看出，两点间的权重要么是$\epsilon$，要么是$0$, 整体$scale$是一样的，丧失了很多数据间相似性信息，因此该方法适用于无权图。

### $k$近邻法

　　利用$K$近邻算法遍历所有的样本点，取每个样本的最近的$k$个点作为近邻，只有和样本最近的$k$个点, $w_{ij}>0$此时$w_{ij}$可以使用高斯相似性函数来计算，否则$w_{ij}=0$.

　　但这样会造成邻接矩阵不是对称的(我们后面会知道，对称矩阵会带来很多计算上的遍历。)因为我的邻近点中有你，你的邻近点中不一定有我。

　　处理成对称矩阵有2种方法。

1) $KNN \ Graph$:  $i,j$中只要有一个点是另一个点的邻近点，则二者的权重都赋值成$s_{ij}$.
$$
w_{ij}=w_{ji}= \begin{cases} 0& {x_i \notin KNN(x_j) \;and \;x_j \notin KNN(x_i)}\\ exp(-\frac{||x_i-x_j||_2^2}{2\sigma^2})& {x_i \in KNN(x_j)\; or\; x_j \in KNN(x_i}) \end{cases}
$$
　具体处理过程为，首先按正常$k$近邻得到邻接矩阵$W$，此时$w_{ij}$可能不等于$w_{ji}$（一者为0，另一者为高斯相似函数值。若相等则不变）。令：
$$
W := (W \ | \  W^T)
$$
2)$Mutual\  KNN \ Graph$, $i,j$必须两个点都是彼此的邻近点，否则都赋值成0。
$$
w_{ij}=w_{ji}= \begin{cases} 0& {x_i \notin KNN(x_j) \;or\;x_j \notin KNN(x_i)}\\ exp(-\frac{||x_i-x_j||_2^2}{2\sigma^2})& {x_i \in KNN(x_j)\; and \; x_j \in KNN(x_i}) \end{cases}
$$
　　具体处理过程，首先按正常$k$近邻得到邻接矩阵$W$，令：
$$
W := W\  \& \  W^T
$$
　　上述的$\&$代表, 如果两个矩阵$W, W^T$相同位置元素值如果相等，则该位置数值保持不变。如果不相等(也就是说有一个不是另一个的邻近点，此时$w=0$)，则令该位置数值为0。

### 全连接法

　　所有点之间的权重都大于0，距离度量可以选择不同的核函数来定义边权重，常用的有多项式核函数，高斯核函数和$Sigmoid$核函数。最常用的是高斯核函数$RBF$，此处使用高斯相似函数$w_{ij}=exp(-\frac{||x_i-x_j||_2^2}{2\sigma^2})$。参数$\sigma$控制了领域的宽度，这和$\epsilon$邻近法中的$\epsilon$作用是相似的。如何理解呢？ 考察样本点$x_j$和其他点的相似性，也就是固定高斯核中心位置在$x_j$，$w_{ij}=exp(-\frac{||x-x_j||_2^2}{2\sigma^2})$, 则$\sigma$越小，高斯函数越尖，领域宽度越小。意味着同一个样本点，代入$x_j$为中心的相似性函数，在不同的$\sigma$取值下，得到的和$x_j$的相似性$w_{ij}$值不一样，且$\sigma$小的相似性低。这意味着两个样本点在$\sigma$大时可能划分到同一个聚类族里，而当$\sigma$小时，可能就不会划分到同一个聚类族里了，间接反映了聚类族(领域)的大小。

　　正常情况下，此时得到的邻接矩阵应该是对称的。不过为了保险，仍然对邻接矩阵进行如下处理：
$$
W:=(W+W^T)/2
$$

## 图拉普拉斯矩阵

　　拉普拉斯矩阵$Laplacians  \ matrices$是用于谱聚类的主要工具。拉普拉斯矩阵具有很多重要的性质。

　　我们假设图$G$是无向带权图，且权重为非负数。当使用矩阵的特征向量时，我们没必要假定图是规范化的，这意味着特征向量$\mathbb{1}$和$a \mathbb{1} ，a!=0$, 是一样的。

### 未规范化的图拉普拉斯矩阵

$$
L = D - W
$$

　　有如下性质：

- 对任意向量$f \in R^n$, 有$f^T L f = \frac{1}{2} \sum_{i,j=1}^n w_{ij} (f_i-f_j)^2$
- $L$是对称且是半正定的。
- $L$的最小特征值为0， 且对应的特征向量为单位向量$\mathbb{1}$, 显然 $L \mathbb{1} = 0 \mathbb{1} $
- $L$有n个非负实数特征值，$0 \leq \lambda_1 \leq \lambda_2 \leq ... \leq \lambda$

　　注意L并不依赖于邻接矩阵对角元素(令W对角元素为$w_{ii}$,则$l_{ii} = \sum w_{ij} - w_{ii} = \sum_{j \neq i} w_{ij}$，$l_{ij} = - w_{ij},j \neq i$)。

​	性质1的证明：
$$
f^TLf = f^TDf - f^TWf = \sum\limits_{i=1}^{n}d_if_i^2 - \sum\limits_{i,j=1}^{n}w_{ij}f_if_j \\
=\frac{1}{2}( \sum\limits_{i=1}^{n}d_if_i^2 - 2 \sum\limits_{i,j=1}^{n}w_{ij}f_if_j + \sum\limits_{j=1}^{n}d_jf_j^2) = \frac{1}{2}\sum\limits_{i,j=1}^{n}w_{ij}(f_i-f_j)^2
$$
　　连通性质：

- 0特征值的重数和图$G$连通分量个数相同。当重数为1时，意味着$G$全连通, 且对应的特征向量为$1$。

- 假设按照连通分量将顶点重新排序，同一连通分量里的顶点在一起，则对应的邻接矩阵$W$为分块对角矩阵，即某个连通分量中的顶点和其他连通分量中的顶点间的连边权重为0。更进一步，拉普拉斯矩阵也是分块对角矩阵，且连通分量个数为分块对角矩阵的个数。
  $$
  L=
  	\left[
  	\begin{array}{c|c}
  	L_1&  \\ \hline 
  	& L_2 \\ \hline 
  	& & ... \\ \hline
  	& & & L_k
  	\end{array}
  	\right]
  $$




　　则$L$的特征向量为$L_i$的特征向量且在其他分块的位置全填充为$0$. 形如，$[0,0,..., 0,1,1,1,1,0,0,...,0]^T$.

每一个$L_i$对应的子图是全连通分量，因此$L_i$的0特征值重数为1，相应的特征向量为$1$. 因此矩阵$L$零特征值的重数为$k$, 每个特征向量是对应连通分量的指示向量。

### 规范化图拉普拉斯矩阵

$$
L_{sym} := D^{-1/2}LD^{-1/2} = I - D^{-1/2}WD^{-1/2} \\
L_{rw} = D^{-1} L = I - D^{-1}W
$$

　　稍后会看到，规范化的拉普拉斯矩阵对应的是图切割问题中的$Ncut$优化目标。而上述未规范化的拉普拉斯矩阵对应的是图切割问题中的$RatioCut$优化目标。稍后会看到，$L_{rw}$的前k个特征向量构成的矩阵是$Ncut$优化目标$min_{ H } H^TLH, st. H^TDH=I$中的最优解。

　　若令$H:=D^{-1/2}F$, 可以将$Ncut$优化目标转成,$min_{F } F^T L_{sym} F, st. F^TF=I$, 改变了约束形式，后者是标准瑞利熵理论形式，其最优解对应$L_{sym}$的前k个特征向量构成的矩阵$F$。若令$H=D^{-1/2}F$, 则$H$为$L_{rw}$的前k个特征向量构成的矩阵。

　　$L_{sym}$和$L_{rw}$性质如下：

- 对任意向量$f \in R^n$, 有$f^T L_{sym} f = \frac{1}{2} \sum_{i,j=1}^n w_{ij} (\frac{f_i}{\sqrt{d_i}}-\frac{f_j}{\sqrt{d_j}})^2$
- $\lambda$是$L_{rw}$的特征值，对应的特征向量为$u$, 当且仅当$\lambda$为$L_{sym}$的特征值，且$w=D^{1/2}u$是对应的特征向量。
- $\lambda$是$L_{rw}$的特征值，对应的特征向量为$u$, 当且仅当$\lambda, u $是广义特征值求解问题$Lu=\lambda Du$的解。



## 无向图切割视角看谱聚类

　　对于无向图G的切割，我们的目标是将图$G(V,E)$切成相互没有连接的$k$个子图, 也即将$V$划分为若干连通分量，$A_1, A_2,...,A_k$，且$A_i \cap A_j=\emptyset，A_1 \cup A_2...\cup A_k=V$ 。对于任意两个不相交的顶点集，定义切割代价为：
$$
W(A,B):=\sum_{i \in A ,j \in B} w_{ij}
$$
　　对于$k$个子图，定义切割代价为：
$$
cut(A_1,A_2,...A_k) = \frac{1}{2}\sum\limits_{i=1}^{k}W(A_i, \overline{A}_i )
$$
　　上述实际上是$one \ vs \ all$思想，每次考察某个顶点集与该顶点集的补，$k$个子图则考虑$k$次。系数$1/2$是因为无向图，同一条边权重，在考察$A$和$\bar{A}$时会被算两次。

　　我们的目标是最小化切割代价，即顶点集内部点的权重大，而顶点集之间顶点连边权重小。


$$
min \  cut(A_1, A_2,..., A_k)
$$
　　但是该优化目标存在问题，必然会导致平凡解。例如对于二划分问题，只会选择权重最小的边，然后将某个点和其余的点分开，导致划分出来的顶点集不平衡。

　　解决上述问题的思路是改变优化目标，加入对顶点集规模的考虑。由此引出$RatioCut$和$NCut$.

　　下面重点讨论$RatioCut$和$NCut$, 以及它们与拉普拉斯矩阵、谱聚类的关系。

### $RatioCut$

　    在$Cut$目标函数中加入对顶点集中顶点个数的考虑。
$$
min \ RatioCut(A_1,A_2,...A_k) = \frac{1}{2}\sum\limits_{i=1}^{k}\frac{W(A_i, \overline{A}_i )}{|A_i|}
$$
　　之所以说这样切割得到的子图规模是平衡的，是因为该优化目标会在顶点集大小差不多大时取到较小的值。特别的,$\sum_{i=1}^k(1/|A_i|)$的最小值会在$|A_i|$全部相等时取到。因此这样的切割得到的顶点集趋向于平衡。

　　那么如何优化该目标函数呢？这里引入$k$个指示向量$h_j =\{h_{1j}, h_{2j},..h_{nj} \}; j =1,2,...k$,  指示向量$h_j$代表第$j$个顶点集的指示向量，它是n维的，n是顶点的个数，也就是该指示向量衡量了所有顶点是否属于该顶点集。
$$
h_{ij}= \begin{cases} 0& { v_i \notin A_j}\\ \frac{1}{\sqrt{|A_j|}}& { v_i \in A_j} \end{cases}  
$$
　　对于某个顶点集$A_i$的指示向量$h_i$, 我们有$||h_i||_2 = 1$(有$|A_i|$个分量数值为$\frac{1}{\sqrt{|A_j|}}$), 且$h_i^Th_j=0$

计算$h_i^T L h_i$有，其中$L$是未规范的图拉普拉斯矩阵:
$$
\begin{align} 
h_i^TLh_i & = \frac{1}{2}\sum\limits_{m=1}\sum\limits_{n=1}w_{mn}(h_{im}-h_{in})^2 \\
& =\frac{1}{2}(\sum\limits_{m \in A_i, n \notin A_i}w_{mn}(\frac{1}{\sqrt{|A_i|}} - 0)^2 +  \sum\limits_{m \notin A_i, n \in A_i}w_{mn}(0 - \frac{1}{\sqrt{|A_i|}} )^2\\
& = \frac{1}{2}(\sum\limits_{m \in A_i, n \notin A_i}w_{mn}\frac{1}{|A_i|} +  \sum\limits_{m \notin A_i, n \in A_i}w_{mn}\frac{1}{|A_i|}\\
& = \frac{1}{2}(cut(A_i, \overline{A}_i) \frac{1}{|A_i|} + cut(\overline{A}_i, A_i) \frac{1}{|A_i|}) \\
& =  \frac{cut(A_i, \overline{A}_i)}{|A_i|} \\
& = RatioCut(A_i, \overline{A}_i) 
\end{align}
$$
　　我们惊喜的发现，这样设计的**【指示向量】将【谱聚类算法】的【拉普拉斯矩阵求解】与【图切割】问题巧妙的联系在一起了**！

　　这意味着求解二划分图切割问题,即最小化$RatioCut(A_i, \overline{A}_i) $可以转成最小化拉普拉斯矩阵的正定型$h_iLh_i$。

　　进一步推广到$k$个子图的切割问题，构造顶点集$A_1, A_2,...,A_k$的指示向量矩阵$H$, 每列对应相应顶点集的指示向量，则：
$$
h_i^T L h_i = (H^TLH)_{ii} \\
RatioCut(A_1,A_2,...A_k) = \sum\limits_{i=1}^{k}h_i^TLh_i = \sum\limits_{i=1}^{k}(H^TLH)_{ii} = tr(H^TLH)
$$
　　也就是说图切割问题优化目标就是最小化$tr(H^TLH)$, 另外根据指示向量的性质，我们有约束$H^TH=I$.加上该约束后，对$min \ tr(H^TLH)$求解得到的结果可以近似认为是图切割问题中的指示向量集合。
$$
\underbrace{arg\;min}_{H \in R^{n*k}}\; tr(H^TLH) \;\; s.t.\;H^TH=I, H \ as \  defined  \  above
$$
　　然而，H矩阵中每一维指示向量都是n维的，取值要么是0,要么是$1/\sqrt{|A_i|}$, 则每个指示向量有$2^n$种，k个指示向量就有$k2^n$ 种$H$。因此该优化问题是np难的，归根结底是因为指示向量的分量取值为离散值决定的。

　　我们考虑松弛约束问题，使H分量的取值可以为任意实数。则转成迹最小化问题：
$$
\underbrace{arg\;min}_{H \in R^{n*k}}\; tr(H^TLH) \;\; s.t.\;H^TH=I
$$
　　这是瑞利熵理论的一种形式。使用拉格朗日方程求解非常容易。考察单独某个指示向量求解问题，
$$
min \ h^T L h  \\
st. ||h||_2 =1
$$

$$
\ell=h^T L h-\lambda(||h||_2-1)=h^T L u - \lambda(h^Th-1)　\\
$$

　　对$h$求导，
$$
\nabla_h \ell=2Lh-2\lambda h = 0 \\
Lh = \lambda h 
$$
　　因此，最优解就是拉普拉斯矩阵的特征向量，该特征向量近似为指示向量，且目标函数最小值也就是该特征向量对应的特征值。对于$k$划分问题，$H$矩阵最优解为L的前k个特征向量构成的矩阵，目标函数值为对应特征值的和（H将L对角化为特征值构成的对角矩阵，相当于对对角元素求和了）。

　　由于约束的松弛，导致得到的优化后的指示向量h对应的H现在不能完全指示各样本的归属，因此一般在得到nxk维度的矩阵H后还需要对每一行进行一次传统的聚类，比如使用K-Means聚类.

### $NCut$

　　$Ncut$换了一种方式考察子图的规模，由于子图样本的个数多并不一定权重就大(即相似性高)，我们切图时基于权重也更符合我们的目标，因此一般来说Ncut切图优于RatioCut切图。
$$
NCut(A_1,A_2,...A_k) = \frac{1}{2}\sum\limits_{i=1}^{k}\frac{W(A_i, \overline{A}_i )}{vol(A_i)}
$$
　　对应的，Ncut切图对指示向量$h_j$做了改进。注意到RatioCut切图的指示向量使用的是$1/\sqrt{|A_j|}$标示样本归属，而Ncut切图使用了子图权重$1/\sqrt{vol(A_j)}$来标示指示向量$h_j$，定义如下:
$$
h_{ij}= \begin{cases} 0& { v_i \notin A_j}\\ \frac{1}{\sqrt{vol(A_j)}}& { v_i \in A_j} \end{cases}
$$
　　此时的指示向量有性质，$h_j^T Dh_j = 1$,  $h_i^Th_j=0,i \neq j$证明：
$$
h_j^TDh_j = \frac{d_1}{vol(A_j)} + \frac{d_2}{vol(A_j)} + ...+ \frac{d_m}{vol(A_j)} \\
=  \frac{d_1+d_2+... + d_m}{vol(A_j)}=\frac{vol(A_j)}{vol(A_j)} = 1 \\
m为顶点集A_j中的元素个数。
$$
　　同样有：
$$
\begin{align} h_i^TLh_i & = \frac{1}{2}\sum\limits_{m=1}\sum\limits_{n=1}w_{mn}(h_{im}-h_{in})^2 \\& =\frac{1}{2}(\sum\limits_{m \in A_i, n \notin A_i}w_{mn}(\frac{1}{\sqrt{vol(A_j)}} - 0)^2 +  \sum\limits_{m \notin A_i, n \in A_i}w_{mn}(0 - \frac{1}{\sqrt{vol(A_j)}} )^2\\& = \frac{1}{2}(\sum\limits_{m \in A_i, n \notin A_i}w_{mn}\frac{1}{vol(A_j)} +  \sum\limits_{m \notin A_i, n \in A_i}w_{mn}\frac{1}{vol(A_j)}\\& = \frac{1}{2}(cut(A_i, \overline{A}_i) \frac{1}{vol(A_j)} + cut(\overline{A}_i, A_i) \frac{1}{vol(A_j)}) \\& =  \frac{cut(A_i, \overline{A}_i)}{vol(A_j)} \\& = NCut(A_i, \overline{A}_i) \end{align}
$$

$$
NCut(A_1,A_2,...A_k) = \sum\limits_{i=1}^{k}h_i^TLh_i = \sum\limits_{i=1}^{k}(H^TLH)_{ii} = tr(H^TLH)
$$

　　则松弛约束后的优化目标为：
$$
\underbrace{arg\;min}_H\; tr(H^TLH) \;\; s.t.\;H^TDH=I
$$
　　此优化目标的最优解为$L_{rw}$的前$k$个特征向量构成的矩阵。对于某个特征向量$h$，可通过拉格朗日方程得到，
$$
L h = \lambda Dh
$$
　　上述也就是$L_{rw}$的特征向量。

　　因为此时约束中包含D， 考虑如下变形化成标准瑞利熵形式：

　　令$H=D^{-1/2}F$, 则$H^TLH = F^TD^{-1/2}LD^{-1/2}F, H^TDH=F^TF = I$

　　此时优化目标变为：
$$
\underbrace{arg\;min}_F\; tr(F^TD^{-1/2}LD^{-1/2}F) \;\; s.t.\;F^TF=I
$$
 　　其中，$D^{-1/2}LD^{-1/2}=L_{sym}$，同样求出$L_{sym}$前k个最小特征值对应的特征向量构成的矩阵$F$, 令$H=D^{-1/2}F$,根据前文的性质，$H$对应$L_{rw}$矩阵的前$k$个特征向量构成的矩阵。

## 谱聚类算法

### 未规范化拉普拉斯矩阵求解（$RatioCut$）

![unnormalized_spectral_clustering](\picture\machine-learning\unnormalized_spectral_clustering.png)

### 规范化拉普拉斯矩阵求解($Ncut$)

1. $L_{rw}$矩阵求特征向量: $Shi \ Algorithm$

![normalized_spectral_clustering_shi](\picture\machine-learning\normalized_spectral_clustering_shi.png)

2. $L_{sym}$矩阵求特征向量: $Ng \ Algorithm$

   ![normalized_spectral_clustering_ng](\picture\machine-learning\normalized_spectral_clustering_ng.png)

　Ng的算法中有一步进行行规范，是因为特征向量为$F=D^{1/2}H$, 若顶点的度很小，导致特征向量分量的值也很小，无法区分出是scale 0 还是scale 1。因此进行规范化。当然如果顶点度真的很小，规范化后特征向量分量会变很大，导致仍然很难区分。一种合理的解释是，对于度很小的顶点，一般都是离群点，这样的点划分错了也无所谓。

## 参考

[A tutorial on spectral clustering](https://arxiv.org/pdf/0711.0189.pdf)

