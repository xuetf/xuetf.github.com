---
title: KDD 21 | 工业界搜推广nlp论文整理
date: 2021-10-07 20:29:32
tags: [推荐系统,paper,KDD]
comments: true
top: 25
---


本文整理了KDD21的Accepted Papers<sup>[1]</sup>中，工业界在搜索、推荐、广告、nlp上的文章。整理的论文列表比较偏个人口味，选取的方式是根据论文作者列表上看是否是公司主导的，但判断比较偏主观，存在漏掉的可能。盘点的方式主要按照公司和方向来划分，排名不计先后顺序。

<!--more-->

# 1. 按照方向分类

主要挑选了一些笔者比较感兴趣的方向，并整理了对应的文章名称。读者可以大致读一下文章名，判断是否和自己的研究方向或工作方向一致，从中选择感兴趣的文章进行精读。

## 1.1 推荐系统

### 1.1.1 样本

涉及到采样、负样本等。

- Google: Bootstrapping for Batch Active Sampling

- Google: Bootstrapping Recommendations at Chrome Web Store

- Alibaba：Real Negatives Matter: Continuous Training with Real Negatives for Delayed Feedback Modeling

### 1.1.2 表征学习

- Google: Learning to Embed Categorical Features without Embedding Tables for Recommendation

- 华为：An Embedding Learning Framework for Numerical Features in CTR Prediction

- 腾讯：Learning Reliable User Representations from Volatile and Sparse Data to Accurately Predict Customer Lifetime Value

- 阿里：Representation Learning for Predicting Customer Orders

### 1.1.3 跨域推荐

- 阿里：Debiasing Learning based Cross-domain Recommendation

- 腾讯：Adversarial Feature Translation for Multi-domain Recommendation

### 1.1.4 纠偏

- 阿里：Contrastive Learning for Debiased Candidate Generation in Large-Scale Recommender Systems

- 阿里：Debiasing Learning based Cross-domain Recommendation

### 1.1.5 图神经网络

- 华为：Dual Graph enhanced Embedding Neural Network for CTR Prediction

- 美团：Signed Graph Neural Network with Latent Groups

- 阿里：DMBGN: Deep Multi-Behavior Graph Networks for Voucher Redemption Rate Prediction

- 百度：MugRep: A Multi-Task Hierarchical Graph Representation Learning Framework for Real Estate Appraisal

### 1.1.6 多任务

- Google：Understanding and Improving Fairness-Accuracy Trade-offs in Multi-Task Learning

- 美团：Modeling the Sequential Dependence among Audience Multi-step Conversions with Multi-task Learning for Customer Acquisition

- 百度：MugRep: A Multi-Task Hierarchical Graph Representation Learning Framework for Real Estate Appraisal

### 1.1.7 多模态

- 阿里：SEMI: A Sequential Multi-Modal Information Transfer Network for E-Commerce Micro-Video Recommendations

### 1.1.8 知识图谱

- Microsoft：Reinforced Anchor Knowledge Graph Generation for News Recommendation Reasoning

### 1.1.9 推荐系统架构

- Facebook：Training Recommender Systems at Scale: Communication-Efficient Model and Data Parallelism

- Facebook：Hierarchical Training: Scaling Deep Recommendation Models on Large CPU Clusters

- 阿里，FleetRec: Large-Scale Recommendation Inference on Hybrid GPU-FPGA Clusters

- 腾讯，Large-Scale Network Embedding in Apache Spark

- Microsoft，On Post-Selection Inference in A/B Testing



## 1.2 搜索

### 1.2.1 向量检索

- 阿里：Embedding-based Product Retrieval in Taobao Search



### 1.2.2 查询/内容理解

- Facebook：Que2Search: Fast and Accurate Query and Document Understanding for Search at Facebook



### 1.2.3 概念图谱

- 阿里巴巴：AliCG: Fine-grained and Evolvable Conceptual Graph Construction for Semantic Search at Alibaba

- 阿里巴巴：AliCoCo2: Commonsense Knowledge Extraction, Representation and Application in E-commerce



### 1.2.4 预训练

- 百度：Pretrained Language Models for Web-scale Retrieval in Baidu Search

- 微软：Domain-Specific Pretraining for Vertical Search: Case Study on Biomedical Literature



### 1.2.5 Query改写/自动补全

- 微软：Diversity driven Query Rewriting in Search Advertising

- 百度：Meta-Learned Spatial-Temporal POI Auto-Completion for the Search Engine at Baidu Maps



### 1.2.6 图神经网络

- 百度：HGAMN: Heterogeneous Graph Attention Matching Network for Multilingual POI Retrieval at Baidu Maps



### 1.2.7 多模态

- Google: Mondegreen: A Post-Processing Solution to Speech Recognition Error Correction for Voice Search Queries

- Facebook：VisRel: Media Search at Scale



### 1.2.8 边缘计算

- 阿里：FIVES: Feature Interaction Via Edge Search for Large-Scale Tabular Data



### 1.2.9 搜索引擎架构

- 百度：Norm Adjusted Proximity Graph for Fast Inner Product Retrieval

- 百度：JIZHI: A Fast and Cost-Effective Model-As-A-Service System for Web-Scale Online Inference at Baidu



## 1.3 广告

这一块文章不是很多，就不细分了。

- Google: Clustering for Private Interest-based Advertising

- 阿里：A Unified Solution to Constrained Bidding in Online Display Advertising

- 阿里：Exploration in Online Advertising Systems with Deep Uncertainty-Aware Learning
- 阿里：Neural Auction: End-to-End Learning of Auction Mechanisms for E-Commerce Advertising
- 阿里：We Know What You Want: An Advertising Strategy Recommender System for Online Advertising



## 1.4 NLP

### 1.4.1 预训练

- 微软：NAS-BERT: Task-Agnostic and Adaptive-Size BERT Compression with Neural Architecture Search

- 阿里：M6: Multi-Modality-to-Multi-Modality Multitask Mega-transformer for Unified Pretraining

- 微软：TUTA: Tree-based Transformers for Generally Structured Table Pre-training

### 1.4.2 命名实体识别

- 微软：Reinforced Iterative Knowledge Distillation for Cross-Lingual Named Entity Recognition

### 1.4.3 少样本学习

- 微软：Generalized Zero-Shot Extreme Multi-label Learning

- 微软：Zero-shot Multi-lingual Interrogative Question Generation for "People Also Ask" at Bing

### 1.4.4 摘要

- 微软：Reinforcing Pretrained Models for Generating Attractive Text Advertisements

### 1.4.5 意图识别

- 阿里：MeLL: Large-scale Extensible User Intent Classification for Dialogue Systems with Meta Lifelong Learning

### 1.4.6 多模态

- 阿里：M6: Multi-Modality-to-Multi-Modality Multitask Mega-transformer for Unified Pretraining



# 2.按照公司分类

## 2.1 Google

- Learning to Embed Categorical Features without Embedding Tables for Recommendation
- NewsEmbed: Modeling News through Pre-trained Document Representations

- Understanding and Improving Fairness-Accuracy Trade-offs in Multi-Task Learning

- Bootstrapping for Batch Active Sampling
- Bootstrapping Recommendations at Chrome Web Store
- Clustering for Private Interest-based Advertising
- Dynamic Language Models for Continuously Evolving Content
- Mondegreen: A Post-Processing Solution to Speech Recognition Error Correction for Voice Search Queries
- On Training Sample Memorization: Lessons from Benchmarking Generative Modeling with a Large-scale Competition

## 2.2 Facebook

- Training Recommender Systems at Scale: Communication-Efficient Model and Data Parallelism
- Preference Amplification in Recommender Systems
- Hierarchical Training: Scaling Deep Recommendation Models on Large CPU Clusters
- Network Experimentation at Scale
- Que2Search: Fast and Accurate Query and Document Understanding for Search at Facebook
- VisRel: Media Search at Scale
- Balancing Consistency and Disparity in Network Alignment

## 2.3 Microsoft

- Generalized Zero-Shot Extreme Multi-label Learning
- Learning Multiple Stock Trading Patterns with Temporal Routing Adaptor and Optimal Transport
- NAS-BERT: Task-Agnostic and Adaptive-Size BERT Compression with Neural Architecture Search
- Reinforced Anchor Knowledge Graph Generation for News Recommendation Reasoning
- Table2Charts: Recommending Charts by Learning Shared Table Representations

- TabularNet: A Neural Network Architecture for Understanding Semantic Structures of Tabular Data

- TUTA: Tree-based Transformers for Generally Structured Table Pre-training

- Contextual Bandit Applications in a Customer Support Bot

- Diversity driven Query Rewriting in Search Advertising

- Domain-Specific Pretraining for Vertical Search: Case Study on Biomedical Literature

- On Post-Selection Inference in A/B Testing

- Reinforced Iterative Knowledge Distillation for Cross-Lingual Named Entity Recognition

- Reinforcing Pretrained Models for Generating Attractive Text Advertisements

- Zero-shot Multi-lingual Interrogative Question Generation for "People Also Ask" at Bing


## 2.4 阿里

- A Unified Solution to Constrained Bidding in Online Display Advertising
- AliCG: Fine-grained and Evolvable Conceptual Graph Construction for Semantic Search at Alibaba
- AliCoCo2: Commonsense Knowledge Extraction, Representation and Application in E-commerce
- Contrastive Learning for Debiased Candidate Generation in Large-Scale Recommender Systems
- Debiasing Learning based Cross-domain Recommendation
- Device-Cloud Collaborative Learning for Recommendation
- Deep Inclusion Relation-aware Network for User Response Prediction at Fliggy
- DMBGN: Deep Multi-Behavior Graph Networks for Voucher Redemption Rate Prediction
- Dual Attentive Sequential Learning for Cross-Domain Click-Through Rate Prediction
- Embedding-based Product Retrieval in Taobao Search
- Exploration in Online Advertising Systems with Deep Uncertainty-Aware Learning
- FIVES: Feature Interaction Via Edge Search for Large-Scale Tabular Data
- FleetRec: Large-Scale Recommendation Inference on Hybrid GPU-FPGA Clusters
- Intention-aware Heterogeneous Graph Attention Networks for Fraud Transactions Detection
- Live-Streaming Fraud Detection: A Heterogeneous Graph Neural Network Approach
- M6: Multi-Modality-to-Multi-Modality Multitask Mega-transformer for Unified Pretraining
- Markdowns in E-Commerce Fresh Retail: A Counterfactual Prediction and Multi-Period Optimization Approach
- MeLL: Large-scale Extensible User Intent Classification for Dialogue Systems with Meta Lifelong Learning
- Multi-Agent Cooperative Bidding Games for Multi-Objective Optimization in e-Commercial Sponsored Search
- Neural Auction: End-to-End Learning of Auction Mechanisms for E-Commerce Advertising
- Real Negatives Matter: Continuous Training with Real Negatives for Delayed Feedback Modeling
- Representation Learning for Predicting Customer Orders
- SEMI: A Sequential Multi-Modal Information Transfer Network for E-Commerce Micro-Video Recommendations
- We Know What You Want: An Advertising Strategy Recommender System for Online Advertising

## 2.5 百度

- Norm Adjusted Proximity Graph for Fast Inner Product Retrieval
- Curriculum Meta-Learning for Next POI Recommendation
- Pretrained Language Models for Web-scale Retrieval in Baidu Search
- HGAMN: Heterogeneous Graph Attention Matching Network for Multilingual POI Retrieval at Baidu Maps
- JIZHI: A Fast and Cost-Effective Model-As-A-Service System for Web-Scale Online Inference at Baidu
- Meta-Learned Spatial-Temporal POI Auto-Completion for the Search Engine at Baidu Maps
- MugRep: A Multi-Task Hierarchical Graph Representation Learning Framework for Real Estate Appraisal
- SSML: Self-Supervised Meta-Learner for En Route Travel Time Estimation at Baidu Maps
- Talent Demand Forecasting with Attentive Neural Sequential Model

## 2.6 腾讯

- Why Attentions May Not Be Interpretable?
-  Adversarial Feature Translation for Multi-domain Recommendation
- Large-Scale Network Embedding in Apache Spark
- Learn to Expand Audience via Meta Hybrid Experts and Critics
- Learning Reliable User Representations from Volatile and Sparse Data to Accurately Predict Customer Lifetime Value

## 2.7 美团

- Modeling the Sequential Dependence among Audience Multi-step Conversions with Multi-task Learning for Customer Acquisition
- User Consumption Intention Prediction in Meituan
- Signed Graph Neural Network with Latent Groups
- A Deep Learning Method for Route and Time Prediction in Food Delivery Service

## 2.8 华为

- An Embedding Learning Framework for Numerical Features in CTR Prediction
- Dual Graph enhanced Embedding Neural Network for CTR Prediction
- Discrete-time Temporal Network Embedding via Implicit Hierarchical Learning
- Retrieval & Interaction Machine for Tabular Data Prediction
- A Multi-Graph Attributed Reinforcement Learning Based Optimization Algorithm for Large-scale Hybrid Flow Shop Scheduling Problem



# 结语

后续笔者会针对感兴趣的文章进行解读。如果大家有感兴趣的文章，也欢迎在公众号后台跟我留言，我会优先挑选大家感兴趣的文章进行解读。当然，如果你有解读好的笔记，也欢迎投稿或交流~~

# 参考

[1] KDD2021 Accepted Papers: https://kdd.org/kdd2021/accepted-papers/index

也欢迎关注我的公众号"**蘑菇先生学习记**"，更快更及时地获取推荐系统前沿进展！

![qr](/picture/qr_sr_code.png)

