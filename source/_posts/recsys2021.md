---
title: Recsys2021 | 推荐系统论文整理和导读
date: 2021-10-07 19:48:31
tags: [推荐系统,paper,Recsys]
comments: true
top: 27
categories: 推荐系统
---

此前整理过KDD21上工业界文章，本文主要整理和分类了Recsys 2021的Research Papers和Reproducibility papers。按照推荐系统的**研究方向**和使用的**推荐技术**来分类，方便大家**快速检索自己感兴趣的文章**。个人认为Recsys这个会议重点不在于"技术味多浓"或者"技术多先进"，而在于经常会涌现很多**新的观点**以及**有意思的研究点**，涵盖推荐系统的各个方面，例如，Recsys 2021涵盖的一些很有意思的研究点包括：


- **推荐系统的信息茧房和回音室**问题的探讨，有4篇文章探讨了社交媒体推荐、音乐推荐和视频推荐中的信息茧房和回音室效应。很少见到在学术会议上专门讨论这样**深刻的问题**，值得一读。
<!--more-->
- **推荐系统评估体系**的探讨，对推荐系统**整个评估体系的梳理**，多个指标间如何做**权衡**等。
- **推荐系统的交互设计**探讨，探讨了美食推荐场景下用户交互设计。关于用户界面/交互设计的推荐系统文章还是很新奇的。
- 推荐系统中的**探索与利用**探讨，例如Google关于用户探索的工作**Values of User Exploration in Recommender Systems**值得一读。
- **对已有工作的探讨和挑战，传统矩阵分解推荐系统和深度学习推荐系统的对比**。例如：何向南老师的NCF工作和MF的对比，继Recsys20被进行对比后, 在Recsys21上又再次被摆上了台面进行对比。
  - **Recsys20**, Rendle S, Krichene W, Zhang L, et al. **Neural collaborative filtering vs. matrix factorization revisited**[C]//Fourteenth ACM Conference on Recommender Systems. 2020: 240-248.
  - **Recsys21**, Anelli V W, Bellogín A, Di Noia T, et al. **Reenvisioning the comparison between Neural Collaborative Filtering and Matrix Factorization**[C]//Fifteenth ACM Conference on Recommender Systems. 2021: 521-529.

还有些研究点也是值得一读的，比如推荐系统中的**冷启动**，**偏差与纠偏**，**序列推荐**，**可解释性，隐私保护**等，这些研究很有意思和启发性**，有助于开拓大家的**研究思路**。

下面主要根据自己读题目或者摘要时的一些判断做的归类，按照**推荐系统研究方向分类**、**推荐技术分类**以及**专门实验性质的可复现型文章分类**，可能存在漏归和错归的情况，请大家多多指正。

# 1.按照推荐系统研究方向分类

## 1.1 信息茧房和回音室

**信息茧房/回音室(echo chamber)/过滤气泡(filter bubble)**，这3个概念类似，在国内外有不同的说法。大致是指使用社交媒体以及带有**算法推荐功能**的资讯类APP，可能会导致我们**只看得到自己感兴趣的、认同的内容**，进而让大家都活在自己的**小世界里**，彼此之间**难以认同和沟通**。关于这部分的概念可参见知乎文章：https://zhuanlan.zhihu.com/p/71844281。有四篇文章探讨了这样的问题。

- **The Dual Echo Chamber: Modeling Social Media Polarization for Interventional Recommending**

  *Tim Donkers and Jürgen Ziegler*

- **I want to break free! Recommending friends from outside the echo chamber**

  *Antonela Tommasel, Juan Manuel Rodriguez, and Daniela Godoy*

- **Follow the guides: disentangling human and algorithmic curation in online music consumption**

  *Quentin Villermet, Jérémie Poiroux, Manuel Moussallam, Thomas Louail, and Camille Roth*

- **An Audit of Misinformation Filter Bubbles on YouTube: Bubble Bursting and Recent Behavior Changes**

  *Matus Tomlein, Branislav Pecher, Jakub Simko, Ivan Srba, Robert Moro, Elena Stefancova, Michal Kompan, Andrea Hrckova, Juraj Podrouzek, and Maria Bielikova*

## 1.2 探索与利用

此次大会在探索与利用上也有很多探讨，例如多臂老虎机、谷歌的新工作，即：用户侧的探索等。

- **Burst-induced Multi-Armed Bandit for Learning Recommendation**

  *Rodrigo Alves, Antoine Ledent, and Marius Kloft*

- **Values of User Exploration in Recommender Systems**

  **Google**, *Minmin Chen, Yuyan Wang, Can Xu, Ya Le, mohit sharma, Lee Richardson, and Ed Chi*

- **Designing Online Advertisements via Bandit and Reinforcement Learning**

  *Yusuke Narita, Shota Yasui, and Kohei Yata*

- **The role of preference consistency, defaults and musical expertise in users’ exploration behavior in a genre exploration recommender**

  *Yu Liang and Martijn C. Willemsen*

- **Top-K Contextual Bandits with Equity of Exposure**

  *Olivier Jeunen and Bart Goethals*

## 1.3 偏差与纠偏
涉及排序学习的纠偏、用户的偏差探索等。

**Debiased Explainable Pairwise Ranking from Implicit Feedback**

*Khalil Damak, Sami Khenissi, and Olfa Nasraoui*

**Mitigating Confounding Bias in Recommendation via Information Bottleneck**

*Dugang Liu, Pengxiang Cheng, Hong Zhu, Zhenhua Dong, Xiuqiang He, Weike Pan, and Zhong Ming*

**User Bias in Beyond-Accuracy Measurement of Recommendation Algorithms**

*Ningxia Wang, and Li Chen*

## 1.4 冷启动

利用图学习、表征学习等做冷启动。

**Cold Start Similar Artists Ranking with Gravity-Inspired Graph Autoencoders**

*Guillaume Salha-Galvan, Romain Hennequin, Benjamin Chapus, Viet-Anh Tran, and Michalis Vazirgiannis*

**Shared Neural Item Representations for Completely Cold Start Problem**

*Ramin Raziperchikolaei, Guannan Liang, and Young-joo Chung*

## 1.5 评估体系

涉及离线或在线评估方法，准确性和多样性等统一指标的设计等。

**Evaluating Off-Policy Evaluation: Sensitivity and Robustness**

*Yuta Saito, Takuma Udagawa, Haruka Kiyohara, Kazuki Mogi, Yusuke Narita, and Kei Tateno*

**Fast Multi-Step Critiquing for VAE-based Recommender Systems**

*Diego Antognini and Boi Faltings*

**Online Evaluation Methods for the Causal Effect of Recommendations**

*Masahiro Sato*

**Towards Unified Metrics for Accuracy and Diversity for Recommender Systems**

*Javier Parapar and Filip Radlinski*

## 1.6 会话/序列推荐

涉及session维度的短序列推荐；使用NLP中常用的Transformers做序列推荐的鸿沟探讨和解决，这个工作本人还挺感兴趣的，后续会精读下！

- **Next-item Recommendations in Short Sessions**

  *Wenzhuo Song, Shoujin Wang, Yan Wang, and SHENGSHENG WANG*
- **Transformers4Rec: Bridging the Gap between NLP and Sequential / Session-Based Recommendation**

  *Gabriel de Souza Pereira Moreira, Sara Rabhi, Jeong Min Lee, Ronay Ak, and Even Oldridge*

- **Denoising User-aware Memory Network for Recommendation**

  *Zhi Bian, Shaojun Zhou, Hao Fu, Qihong Yang, Zhenqi Sun, Junjie Tang, Guiquan Liu, kaikui liu, and Xiaolong Li*
- **Large-Scale Modeling of Mobile User Click Behaviors Using Deep Learning**

  *Xin Zhou and Yang Li*

## 1.7 隐私保护

结合联邦学习做隐私保护等。

- **Privacy Preserving Collaborative Filtering by Distributed Mediation**

  *Alon Ben Horin, and Tamir Tassa*

- **Stronger Privacy for Federated Collaborative Filtering With Implicit Feedback**

  *Lorenzo Minto, Moritz Haller, Ben Livshits, and Hamed Haddadi*

## 1.8 对抗与攻击

**Black-Box Attacks on Sequential Recommenders via Data-Free Model Extraction**

*Zhenrui Yue, Zhankui He, Huimin Zeng, and Julian McAuley*

## 1.9 对话推荐系统

**Large-scale Interactive Conversational Recommendation System**

*Ali Montazeralghaem, James Allan, and Philip S. Thomas*

## 1.10 可解释性推荐

**EX3: Explainable Attribute-aware Item-set Recommendations**

*Yikun Xian, Tong Zhao, Jin Li, Jim Chan, Andrey Kan, Jun Ma, Xin Luna Dong, Christos Faloutsos, George Karypis, S. Muthukrishnan, and Yongfeng Zhang*

## 1.11 跨域推荐

**Towards Source-Aligned Variational Models for Cross-Domain Recommendation**

*Aghiles Salah, Thanh Binh Tran, and Hady Lauw*

## 1.12 基于视觉的推荐

利用视觉信息做推荐。

- **Semi-Supervised Visual Representation Learning for Fashion Compatibility**

*Ambareesh Revanur, Vijay Kumar, and Deepthi Sharma*

- **Tops, Bottoms, and Shoes: Building Capsule Wardrobes via Cross-Attention Tensor Network**

*Huiyuan Chen, Yusan Lin, Fei Wang, and Hao Yang*

## 1.13 组推荐/用户物品分层推荐

- **Local Factor Models for Large-Scale Inductive Recommendation**

  *Longqi Yang, Tobias Schnabel, Paul N. Bennett, and Susan Dumais*
- **Learning to Represent Human Motives for Goal-directed Web Browsing**

  *Jyun-Yu Jiang, Chia-Jung Lee, Longqi Yang, Bahareh Sarrafzadeh, Brent Hecht, Jaime Teevan*

## 1.14 推荐系统交互设计

探讨了美食场景下，多用户意图的推荐系统的交互设计。

**“Serving Each User”: Supporting Different Eating Goals Through a Multi-List Recommender Interface**

*Alain Starke, Edis Asotic, and Christoph Trattner*


# 2. 按照推荐技术分类

涉及传统协同过滤、度量学习的迭代；新兴的图学习技术、联邦学习技术、强化学习技术等的探索。

## 2.1 协同过滤

**Matrix Factorization for Collaborative Filtering Is Just Solving an Adjoint Latent Dirichlet Allocation Model After All**

*Florian Wilhelm*

**Negative Interactions for Improved Collaborative-Filtering: Don’t go Deeper, go Higher**
*Harald Steck and Dawen Liang*

**ProtoCF: Prototypical Collaborative Filtering for Few-shot Item Recommendation**

*Aravind Sankar, Junting Wang, Adit Krishnan, and Hari Sundaram*

## 2.2 图学习

知识图谱的应用以及图嵌入技术和上下文感知的表征技术的融合，这两个工作个人都挺感兴趣。

- **Sparse Feature Factorization for Recommender Systems with Knowledge Graphs**

*Antonio Ferrara, Vito Walter Anelli, Tommaso Di Noia, and Alberto Carlo Maria Mancino*

- **Together is Better: Hybrid Recommendations Combining Graph Embeddings and Contextualized Word Representations**

*Marco Polignano, Cataldo Musto, Marco de Gemmis, Pasquale Lops, and Giovanni Semeraro*

## 2.3 强化学习

- **Partially Observable Reinforcement Learning for Dialog-based Interactive Recommendation**

  *Yaxiong Wu, Craig Macdonald, and Iadh Ounis,*

- **Pessimistic Reward Models for Off-Policy Learning in Recommendation**

  *Olivier Jeunen and Bart Goethals*

## 2.4 度量学习

- **Hierarchical Latent Relation Modeling for Collaborative Metric Learning**

  *Viet-Anh Tran, Guillaume Salha-Galvan, Romain Hennequin, and Manuel Moussallam*

## 2.5 联邦学习

- **A Payload Optimization Method for Federated Recommender Systems**

  *Farwa K. Khan, Adrian Flanagan, Kuan Eeik Tan, Zareen Alamgir, and Muhammad Ammad-ud-din*

- **Stronger Privacy for Federated Collaborative Filtering With Implicit Feedback**

  Lorenzo Minto, Moritz Haller, Ben Livshits, and Hamed Haddadi


## 2.6 架构/训练/优化

涉及训练、优化、检索、实时流等。

- **cDLRM: Look Ahead Caching for Scalable Training of Recommendation Models**

  *Keshav Balasubramanian, Abdulla Alshabanah, Joshua D Choe, and Murali Annavaram*

- **Reverse Maximum Inner Product Search: How to efficiently find users who would like to buy my item?**

  *Daichi Amagata and Takahiro Hara*
- **Page-level Optimization of e-Commerce Item Recommendations**
  *Chieh Lo, Hongliang Yu, Xin Yin, Krutika Shetty, Changchen He, Kathy Hu, Justin M Platz, Adam Ilardi, and Sriganesh Madhvanath*

- **Accordion: A Trainable Simulator for Long-Term Interactive Systems**

  *James McInerney, Ehtsham Elahi, Justin Basilico, Yves Raimond, and Tony Jebara*

- **Information Interactions in Outcome Prediction: Quantification and Interpretation using Stochastic Block Models**

  *Gaël Poux-Médard, Julien Velcin, and Sabine Loudcher*

- **Learning An Adaptive Meta Model-Generator for Incrementally Updating Recommender Systems**

  *Danni Peng, Sinno Jialin Pan, Jie Zhang, and Anxiang Zeng*

- **Recommendation on Live-Streaming Platforms: Dynamic Availability and Repeat Consumption**

*Jeremie Rappaz, Julian McAuley, and Karl Aberer*


# 3. 实验性质的文章

Reproducibility papers可复现实验性质的文章，共3篇。分别探索了：序列推荐中的**采样评估策略**；对话推荐系统中**生成式和检索式的方法对比**；**神经网络**推荐系统和**矩阵分解**推荐系统的对比。

- **A Case Study on Sampling Strategies for Evaluating Neural Sequential Item Recommendation Models**

  *by Alexander Dallmann, Daniel Zoller, Andreas Hotho (Data Science Chair, University of Würzburg, Würzburg, Germany)*

- **Generation-based vs. Retrieval-based Conversational Recommendation: A User-Centric Comparison**

  *by Ahtsham Manzoor and Dietmar Jannach (University of Klagenfurt, Klagenfurt, Austria)*

- **Reenvisioning the comparison between Neural Collaborative Filtering and Matrix Factorization**

  *by Vito Walter Anelli (Polytechnic University of Bari, Bari, Italy), Alejandro Bellogin (Information Retrieval Group, Universidad Autonoma de Madrid, Madrid, Spain), Tommaso Di Noia Polytechnic (University of Bari, Bari, Italy), and Claudio Pomo (Polytechnic University of Bari, Bari, Italy)



# 总结

通过论文的整理和分类，笔者也发现了一些自己感兴趣的研究点，比如：推荐系统的回音室效应探讨文章；Transformers在序列推荐和NLP序列表征中的鸿沟和解决文章：Transformers4Rec；图嵌入表征和上下文感知表征的融合文章；NCF和MF的实验对比文章；谷歌的用户探索文章等。希望读者也能够发现自己感兴趣的文章。下期分享见！

也欢迎关注我的公众号"**蘑菇先生学习记**"，更快更及时地获取推荐系统前沿进展！

![qr](/picture/qr_sr_code.png)




