---
title: Variational Inference:A Review for Statisticians读书笔记
date: 2018-09-10 09:56:43
tags: [变分推断,机器学习,Paper]
categories: 自然语言处理
---


现代统计学核心问题之一是近似复杂的概率密度。这个问题在贝叶斯统计中尤其重要。贝叶斯统计框架下，所有的推断问题都是要求未知变量的后验概率。而后验概率通常是很难计算的，因此需要相应的算法来近似它。本文主要是阅读David M.Blei 2018发表的论文《Variational Inference: A Review for Statisticians》后的笔记。主要总结其中3种变分推断情况以及对应的优化方法。
<!--more-->


## 一般情况

应用ELBO和Mean Field求解的一般步骤。

- 1) 写出隐变量和观测数据的联合概率分布：$p(X,Z)$。可以画出概率图模型，根据依赖关系写。

- 2) 根据Mean Field写出$q(Z)$公式。$q(Z)=\prod_{j} q_j(z_j)$。具体求解过程中，注意global和local。

- 3) 将1)中联合概率$P(X,Z)$和2)中$q(Z)$代入到ELBO公式：$E_q[log\ p(X,Z)]-E_q[log \ q(Z)]$，ELBO公式中期望是关于$q(Z)$的。对ELBO进行推导，代入$q(Z)$。可以得出一个**重要的结论**1：
  $$
  q^{\*}_j(z_j) \propto exp(E_{-j}[log P(z_j|Z_{-j},X)]) \\
  or \\
  q^{\*}_j(z_j) \propto exp(E_{-j}[log P(X,Z)])
  $$
  $-j$是对除了$z_j$之外的其它隐变量求期望。上述期望最后的形式只会包含变分参数。

- 4) 应用3中的公式，求得不同变分参数的迭代公式。核心步骤。即需要求$E_{-j}[log P(X,Z)]$。注意，3)中是整个$q_j(z_j)$的最优分布满足该结论。如果通过推导，可以得出$q_j(z_j)$的形式，如：某种指数分布族形式，那么可以直接将$q_j(z_j|\phi_j)$和$q^{\*}_j(z_j)$转成指数分布族的形式，并令二者自然参数相等，即可得到变分参数$\phi_j$的迭代公式。或者有的时候，变分参数和分布之间就有某种转换关系，如Categorical中，$q(c_i=k)=\phi_{ik}$，就可以直接更新参数$\phi_{ik}$。 否则其他情况下，只能更新$q_j(z_j)$整个分布。

- 5) 使用CAVI（Coordinate-ascent variational inference）算法迭代更新参数。

  ​

  ​

## 指数分布族情况

如果某个**隐变量的完全后验概率**（complete conditionals）属于指数分布族，那么有更简便的更新方法。

隐变量$z_j$的完全后验概率为：
$$
p(z_j|Z_{-j},X)
$$
$z_{-j}$是除了$z_j$隐变量之外的其余隐变量，但通常可以简化。一般需要根据图模型找到$z_j$的依赖节点，子节点、父节点、co-parent节点。

如果该完全后验概率是指数族形式：
$$
p(z_j|Z_{-j},X) = h(z_j)exp\left(\eta_j(Z_{-j},X)^T t(z_j) - A(\eta_j(Z_{-j},X)) \right)
$$
那么根据Mean Field, 上述的结论，$q^{\*}_j(z_j) \propto exp(E_{-j}[log P(z_j|Z_{-j},X)])$。可以得出：
$$
q^{\*}_j(z_j) \propto exp(E_{-j}[log P(z_j|Z_{-j},X)])  \\
= exp \left(log h(z_j)+ E_{-j}[\eta_j(Z_{-j},X)]^Tt(z_j) - E_{-j}[A(\eta_j(Z_{-j},X)) ] \right) \\
\propto h(z_j)exp(E_{-j}[\eta_j(Z_{-j},X)]^Tt(z_j) )
$$
可以看出，$q_j(z_j)$的最优分布和完全后验概率属于同一指数分布族，形式完全一样，包括自然参数和充分统计量。

因此，假设$\lambda_j$是$q_j(z_j)$的变分**自然**参数。那么更新公式如下，**结论2如下**：
$$
\lambda_j = \mathbb{E}_{-j}[\eta_j(z_{-j},x)]
$$
即，**某个变量的自然参数迭代公式等于完全后验概率的自然参数关于其他变量的期望**。

可以看出，上述结论唯一要求：**隐变量的完全后验概率（complete conditionals）属于指数分布族。**

不需要关于共轭的条件。

**如果一开始假设了$q_j(z_j)$的分布，那么必须保证$q_j(z_j)$的分布和隐变量的后验概率的分布必须属于同一指数族形式**，即，将$q_j(z_j)$转成指数分布族之后，自然参数和充分统计量和隐变量后验概率分布的自然参数和充分统计量对应一致。这样的话，可以令二者自然参数相等，来单独更新$q_j(z_j)$的变分参数。即，上述更新公式为：$\eta(\lambda_j) = \mathbb{E}_{-j}[\eta_j(z_{-j},x)]$。那么需要转成传统参数的更新，$\lambda_j = \eta^{-1}(\lambda_j)$。

## 条件共轭情况

一种关于指数分布族模型的特殊情况是条件共轭(conditionally conjugate models with local and global variables)。条件共轭中的全局变量通常是指"参数"(先验)，而局部变量通常是针对每个数据点的"隐变量"(数据似然)。全局变量影响所有的数据，局部变量只影响单个训练数据。联合概率密度：
$$
p(\beta,z,x)=p(\beta)\prod_{i=1}^n p(z_i,x_i|\beta)
$$
其中，$\beta$是全局变量，$z_i$是局部变量。

上式必须保证每个变量($\beta,z$)的完全后验概率分布都是指数分布族形式，即上文提到的。

接着定义似然分布和先验分布构成条件共轭。假设数据似然属于指数分布族：
$$
p(z_i,x_i|\beta)=h(z_i,x_i)exp(\beta^T t(z_i,x_i)- A(\beta))
$$
那么，为了使得先验为共轭分布，先验$p(\beta)$一种构造方式如下：
$$
p(\beta) = h(\beta)exp(\alpha^T [\beta, -A(\beta)]-A(\alpha))
$$
$\alpha$是先验分布的自然参数，$\alpha=[\alpha_1,\alpha_2]^T$。充分统计量使用$[\beta, -A(\beta)]$，即$\beta$和Log Normalizer。$\alpha_1$和$\beta$的维度一样，$\alpha_2$是一维的。此时可以证明，后验$p(\beta|z_i,x_i)$和$p(\beta)$分布一致。证明如下：
$$
p(\beta|z_i,x_i) = p(\beta)\prod_{i=1}^n p(z_i,x_i|\beta) \\
=h(\beta)exp(\alpha^T [\beta, -A(\beta)]-A(\alpha)) \prod_{i=1}^n \left(h(z_i,x_i)exp(\beta^T t(z_i,x_i)- A(\beta))\right) \\
\propto  h(\beta)exp(\alpha_1\beta -\alpha_2A(\beta)-A(\alpha_1,\alpha_2)) exp(\beta^T \sum_{i=1}^n t(z_i,x_i)-nA(\beta))\\
\propto h(\beta)exp([\alpha_1+\sum_{i=1}^n t(z_i,x_i), \alpha_2+n]^T [\beta, -A(\beta)]) \\
= h(\beta)exp([\hat{\alpha}_1, \hat{\alpha_2}]^T [\beta, -A(\beta)])=p(\beta|\hat{\alpha})\\
$$
可以看出，后验分布的充分统计量和先验分布一样。只需要令二者自然参数一致，即可得到，**结论3**：
$$
\hat{\alpha}_1 = \alpha_1+\sum_{i=1}^n t(z_i,x_i) \\
\hat{\alpha}_2  =\alpha_2 + n
$$
由于我们假定每个隐变量的后验概率分布$p(z_j|Z_{-j},X)$都是指数分布族，因此，根据上文提到的结论：
$$
\lambda_j = \mathbb{E}_{-j}[\eta_j(z_{-j},x)]
$$
其中，$\lambda_j$是$q_j(z_j|\lambda_j)$的变分**自然**参数。

根据该结论，令$q_i(z_i|\varphi_i)$和$q(\beta|\lambda)$，

则，对于局部变量$\varphi_i$有：
$$
\varphi_i = \mathbb{E}_\lambda[\eta(\beta,x_i)]
$$
对于全局变量$\lambda$，由于上述条件共轭推出的结论，有：
$$
\lambda = [\alpha_1+\sum_{i=1}^n \mathbb{E}_{\varphi_i}[t(z_i,x_i)],  \alpha_2+n]^T
$$

## 总结

3大结论：

- ELBO+Mean Field Family

  隐变量分布更新公式如下：
  $$
  q^{\*}_j(z_j) \propto exp(E_{-j}[log P(X,Z)])
  $$

- Complete Conditional in Exponential Family

  变分自然参数更新公式如下：
  $$
  \lambda_j = \mathbb{E}_{-j}[\eta_j(z_{-j},x)]
  $$

- Conditional Conjugacy

  全局参数更新公式如下：
  $$
  \lambda = [\alpha_1+\sum_{i=1}^n \mathbb{E}_{\varphi_i}[t(z_i,x_i)],  \alpha_2+n]^T
  $$


## 引用

[Variational Inference: A Review for Statisticians](https://arxiv.org/pdf/1601.00670.pdf)