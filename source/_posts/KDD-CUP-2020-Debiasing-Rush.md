---
title: KDD CUP 2020之Debiasing赛道方案
date: 2020-06-17 21:03:17
tags: [kddcup2020,Debiasing,推荐系统,GNN,深度学习]
comments: true
categories: GNN
top: 20
---

此次比赛是典型的序列推荐场景中的纠偏问题，即：debiasing of next-item-prediction。模型构建的过程中要重点考虑行为序列和蕴含在序列中的时间信息，位置信息和长短期偏好等。为此，本文提出了一种融合传统协同过滤方法和图神经网络方法的多路召回模型以及集成了GBDT和DIN的排序模型的方案。该方案遵循了推荐系统的主流架构，即召回+粗排+精排。召回方案主要包括了多种改进后的协同过滤方法，即：user-cf、item-cf、swing、bi-graph，以及改进的基于序列的图神经网络SR-GNN[1]方法。这些改进的召回方法能够有效进行数据的纠偏。对每种召回方法，粗排阶段会基于物品的流行度等因素对每个用户的推荐结果进行初步重排，然后将不同方法的Recall的结果初步融合起来。精排方案主要采用了GBDT和DIN[4]方法，会重点挖掘召回特征，内容特征和ID类特征。最后通过集成GBDT和DIN产生排序结果。最终，我们团队**Rush**的方案在Track B中，**full指标第3名，half指标第10名。**

方案解析也可参考我的知乎：https://zhuanlan.zhihu.com/p/149061129

目前代码已开源：https://github.com/xuetf/KDD_CUP_2020_Debiasing_Rush

<!--more-->

# 赛题解析

赛题介绍：[KDD Cup 2020 Challenges for Modern E-Commerce Platform: Debiasing](https://tianchi.aliyun.com/competition/entrance/231785/information)
主要包括了4个数据集：

- underexpose_user_feat.csv: **用户的特征**，uid, age, gender, city。缺失值非常多。
- underexpose_item_feat.csv: **物品的特征**，iid, 128维图片向量+128维文字向量。
- underexpose_train_click-T.csv:  uid, iid, time, **训练集**，记录了用户历史点击行为。
- underexpose_test_click-T.csv: uid, iid, time, **测试集**，记录了待预测用户的历史点击行为。赛题方  还给出了要预测的用户下一次发生点击行为时的时间，即：underexpose_test_qtime-T.csv

目标是基于用户历史点击行为，来预测下一次用户会点击的item，即**next-item prediction**。

根据赛题介绍和对数据集的观察，可以推测主办方是从全量数据里头随机采样部分用户，将这些用户的点击数据作为赛题的数据。在进行数据划分的时候，选取了部分用户的数据作为测试集test，其他用户的数据作为训练集train。对于测试集，将每个用户行为序列的最后一次交互item作为线上测试answer，行为序列去除掉最后一个交互item以外的作为test用户的历史行为数据公开给我们，同时将answer中的user id和query time也公开给我们，即，test_q_time。具体如下图所示：

![线上数据划分](/picture/machine-learning/data_preview.png)

显然，这是典型的序列推荐场景，即**next-item-prediction**。模型构建的过程中要**重点考虑行为序列和蕴含在序列中的时间信息，位置信息和长短期偏好等**。

为了保证线上线下数据分布的一致性，验证集划分思路可参考线上数据的划分方式。即，利用线上train训练集进行划分，从train数据集中随机采样1600个用户，将这1600个用户的最后一次交互item作为验证集answer，其它数据作为验证集用户的历史行为数据。具体如下图所示：

![验证集划分](/picture/machine-learning/tr_val_split.png)

这样的划分，保证了离线环境和线上环境的一致性。上述操作对每个phase都会进行这样的划分过程。

# 数据分析

几个重要的数据分析观察和结论如下：

- 经过统计分析，每个阶段的时间范围一致，不同阶段按照时间推移，且不同阶段的时间重叠部分占到了阶段时间区间的3/4，因此会出现当前阶段记录不完全的情况，所以训练模型时需要考虑使用联合多个phase的全量数据训练模型。**推测可能是线上打点日志系统的延迟上报，或者主办方对每个阶段的数据，都是从某个较大的时间区间内通过滑动窗口的方式随机采样得到的，因此样本存在较大的时间重叠。**

- 经过验证集上的统计，每个用户的最后一次点击有99%以上是在当前阶段出现过的item，因此利用全量数据时需要将不属于当前phase的item过滤掉，防止item的穿越。

- 一条相同的点击数据可能会分布在各个阶段之中，重复率占比非常高，因此需要对记录进行**去重处理**。

- item出现的次数呈现典型的长尾分布，在重排阶段需要挖掘长尾物品，如结合物品出现的频次进行纠偏。

  ![item_count](/picture/machine-learning/item_count.png)

- 其它的一些分析包括，最后一次点击和倒二次点击之间的内容相似性、基于w2v嵌入的行为相似性等分析。不一一列举。

# 方案

我们的方案遵循了推荐系统的主流架构，即召回+粗排+精排。召回方案主要包括了多种改进后的协同过滤方法，即：**user-cf**、**item-cf**、**swing**、**bi-graph**。以及改进的基于序列的**图神经网络**SR-GNN方法。对每种召回方法，粗排阶段会基于物品的流行度等因素对每个用户的推荐结果进行初步重排，然后将不同方法的Recall的结果初步融合起来。精排方案主要采用了GBDT和DIN方法，会重点挖掘召回特征，内容特征和ID类特征。最终产生的结果是**GBDT**和**DIN**的集成。

## 召回方案

### 召回训练集构造

经过数据分析，我们发现不同阶段的数据存在明显的交叉，说明了不同阶段之间不存在明确的时间间隔。因此，我们希望充分利用所有阶段的数据。但是直接利用所有阶段的数据会造成**非常严重的数据穿越问题**。为了保证数据不穿越，我们对全量数据做了进一步的筛选。这是本方案的**key points**之一。具体包括两点：

- 1)  对每个用户，根据测试集中的q-time，将q-time之后的数据过滤掉，防止user的行为穿越。

- 2)  对1) 中过滤后的数据，进一步，把不在当前阶段出现的item的行为数据过滤掉，防止item穿越。
```python
def get_whole_phase_click(all_click, click_q_time):
    '''
    get train data for target phase from whole click
    :param all_click: the click data of target phase
    :param click_q_time: the infer q_time of target phase
    :return: the filtered whole click data for target phase
    '''
    whole_click = get_whole_click()

    phase_item_ids = set(all_click['item_id'].unique())
    pred_user_time_dict = dict(zip(click_q_time['user_id'], click_q_time['time']))

    def group_apply_func(group_df):
        u = group_df['user_id'].iloc[0]
        if u in pred_user_time_dict:
            u_time = pred_user_time_dict[u]
            group_df = group_df[group_df['time'] <= u_time]
        return group_df

    phase_whole_click = whole_click.groupby('user_id', 	group_keys=False).apply(group_apply_func)
    print(phase_whole_click.head())
    print('group done')
    # filter-out the items that not in this phase
    phase_whole_click = phase_whole_click[phase_whole_click['item_id'].isin(phase_item_ids)]
    return phase_whole_click
```

对每个阶段，经过上述步骤后得到筛选后的针对该阶段的全量训练数据，会作为多路召回模型的输入进行训练和召回。

### 多路召回

多路召回包括了4种改进的协同过滤方法以及改进的图神经网络SR-GNN方法。

#### Item-CF

参考item-cf [7, 8]的实现，考虑了交互时间信息，方向信息、物品流行度、用户活跃度等因素对模型的影响对模型的影响。
$$
sim(i,j) = \frac{1}{\sqrt{|U_i||U_j|}} \sum_{u \in U_i \cap U_j} \frac{\left(exp(-\alpha * |t_i - t_j|)\right) \times (c \cdot \beta^{|l_i-l_j|-1})}{\log(1+|I_u|)} \tag{1}
$$
其中，

- $\left(exp(-\alpha * |t_i - t_j|)\right)$考察了交互时间差距因素的影响，$\alpha=15000$
- $(c \cdot \beta^{|l_i-l_j|-1})$考虑交互方向的影响，$\beta=0.8$；正向时，即$l_i > l_j$时，$c=1$，否则，即，反向时，$c=0.7$
- $\sqrt{|U_i||U_j|}$考虑了物品流行度的影响，越流行的商品，协同信号越弱。$U_i$即为交互过物品$i$的用户。
- $log(1+|I_u|)$考虑了用户活跃度的影响，越活跃的用户，协同信号越弱，$I_u$是用户$u$的交互过的物品。

上述改进能够有效进行纠偏。

```python
def get_time_dir_aware_sim_item(df):
    user_item_time_dict = get_user_item_time_dict(df)

    sim_item = {}
    item_cnt = defaultdict(int)
    for user, item_time_list in tqdm(user_item_time_dict.items()):
        for loc_1, (i, i_time) in enumerate(item_time_list):
            item_cnt[i] += 1
            sim_item.setdefault(i, {})
            for loc_2, (relate_item, related_time) in enumerate(item_time_list):
                if i == relate_item:
                    continue
                loc_alpha = 1.0 if loc_2 > loc_1 else 0.7
                loc_weight = loc_alpha * (0.8 ** (np.abs(loc_2 - loc_1) - 1))
                time_weight = np.exp(-15000 * np.abs(i_time - related_time))

                sim_item[i].setdefault(relate_item, 0)
                sim_item[i][relate_item] += loc_weight * time_weight / math.log(1 + len(item_time_list))

    sim_item_corr = sim_item.copy()
    for i, related_items in tqdm(sim_item.items()):
        for j, cij in related_items.items():
            sim_item_corr[i][j] = cij / math.sqrt(item_cnt[i] * item_cnt[j])

    return sim_item_corr, user_item_time_dict
```



#### User-CF

在原始user-cf基础上考虑了用户活跃度、物品的流行度因素。
$$
sim(u,v) = \frac{1}{\sqrt{|I_u||I_v|}} \sum_{i \in I_u \cap I_v} \frac{1}{log(1+|U_i|)}  \tag{2}
$$

```python
def get_sim_user(df):
    # user_min_time_dict = get_user_min_time_dict(df, user_col, item_col, time_col) # user first time
    # history
    user_item_time_dict = get_user_item_time_dict(df)
    # item, [u1, u2, ...,]
    item_user_time_dict = get_item_user_time_dict(df)

    sim_user = {}
    user_cnt = defaultdict(int)
    for item, user_time_list in tqdm(item_user_time_dict.items()):
        num_users = len(user_time_list)
        for u, t in user_time_list:
            user_cnt[u] += 1
            sim_user.setdefault(u, {})
            for relate_user, relate_t in user_time_list:
                # time_diff_relate_u = 1.0/(1.0+10000*abs(relate_t-t))
                if u == relate_user:
                    continue
                sim_user[u].setdefault(relate_user, 0)
                weight = 1.0
                sim_user[u][relate_user] += weight / math.log(1 + num_users)

    sim_user_corr = sim_user.copy()
    for u, related_users in tqdm(sim_user.items()):
        for v, cuv in related_users.items():
            sim_user_corr[u][v] = cuv / math.sqrt(user_cnt[u] * user_cnt[v])

    return sim_user_corr, user_item_time_dict
```



#### Swing

基于图结构的推荐算法Swing [9]，将物品的流行度因素也考虑进去。
$$
Sim(i,j)=\frac{1}{\sqrt{|U_i||U_j|}} \sum_{u \in U_i \cap U_j} \sum_{v \in U_i \cap U_j} \frac{1}{\alpha+|I_u \cap I_v|} \tag{3}
$$

```python
def swing(df, user_col='user_id', item_col='item_id', time_col='time'):
    # 1. item, (u1,t1), (u2, t2).....
    item_user_df = df.sort_values(by=[item_col, time_col])
    item_user_df = item_user_df.groupby(item_col).apply(
        lambda group: make_user_time_tuple(group, user_col, item_col, time_col)).reset_index().rename(
        columns={0: 'user_id_time_list'})
    item_user_time_dict = dict(zip(item_user_df[item_col], item_user_df['user_id_time_list']))

    user_item_time_dict = defaultdict(list)
    # 2. ((u1, u2), i1, d12)
    u_u_cnt = defaultdict(list)
    item_cnt = defaultdict(int)
    for item, user_time_list in tqdm(item_user_time_dict.items()):
        for u, u_time in user_time_list:
            # just record
            item_cnt[item] += 1
            user_item_time_dict[u].append((item, u_time))

            for relate_u, relate_u_time in user_time_list:
                if relate_u == u:
                    continue

                key = (u, relate_u) if u <= relate_u else (relate_u, u)
                u_u_cnt[key].append((item, np.abs(u_time - relate_u_time)))

    # 3. (i1,i2), sim
    sim_item = {}
    alpha = 5.0
    for u_u, co_item_times in u_u_cnt.items():
        num_co_items = len(co_item_times)
        for i, i_time_diff in co_item_times:
            sim_item.setdefault(i, {})
            for j, j_time_diff in co_item_times:
                if j == i:
                    continue
                weight = 1.0  # np.exp(-15000*(i_time_diff + j_time_diff))
                sim_item[i][j] = sim_item[i].setdefault(j, 0.) + weight / (alpha + num_co_items)
    # 4. norm by item count
    sim_item_corr = sim_item.copy()
    for i, related_items in sim_item.items():
        for j, cij in related_items.items():
            sim_item_corr[i][j] = cij / math.sqrt(item_cnt[i] * item_cnt[j])

    return sim_item_corr, user_item_time_dict
```

#### Bi-Graph

Bi-Graph [3, 10] 核心思想是将user和item看做二分图中的两个集合，即：用户集合和物品集合，通过不同集合的关系进行单模式投影得到item侧的物品之间的相似性度量。改进方式：将时间因素、商品热门度、用户活跃度三因素考虑进去。
$$
Sim(i,j)= \sum_{u \in U_i} \sum_{j \in I_u} \frac{\left(\exp(-\alpha * |t_i - t_j|)\right)}{\log(|I_u|+1) \cdot \log(|U_j|+1)} \tag{4}
$$

```python
def get_bi_sim_item(df):
    item_user_time_dict = get_item_user_time_dict(df,)
    user_item_time_dict = get_user_item_time_dict(df)

    item_cnt = defaultdict(int)
    for user, item_times in tqdm(user_item_time_dict.items()):
        for i, t in item_times:
            item_cnt[i] += 1

    sim_item = {}

    for item, user_time_lists in tqdm(item_user_time_dict.items()):

        sim_item.setdefault(item, {})

        for u, item_time in user_time_lists:

            tmp_len = len(user_item_time_dict[u])

            for relate_item, related_time in user_item_time_dict[u]:
                sim_item[item].setdefault(relate_item, 0)
                weight = np.exp(-15000 * np.abs(related_time - item_time))
                sim_item[item][relate_item] += weight / (math.log(len(user_time_lists) + 1) * math.log(tmp_len + 1))

    return sim_item, user_item_time_dict
```



#### SR-GNN

SR-GNN [1] 是将GNN用于序列推荐的一种模型，原论文的方法在多个数据集上都表现出较好的性能。SR-GNN通过GGNN能够捕捉序列中不同item之间的多阶关系，同时会综合考虑序列的长短期偏好，尤其是短期的最后一次交互item，天然适用于该比赛的场景。但是直接使用原始论文开源的代码[13]，在我们的比赛场景中，召回效果不佳，还不如单个CF方法来的好，因此需要进行改进。

我们将用户的行为记录按时间戳排序，然后对用户序列进行数据增强操作，得到增强后的行为序列后，使用改进的SR-GNN实施召回。具体改进如下：

#####  **嵌入初始化**

由于训练样本较少，难以对物品嵌入矩阵进行充分的学习，因此不宜使用随机初始化。考虑到比赛提供的数据中包含了物品特征，为此我们使用物品的文本描述和图片描述向量（共256维）对嵌入矩阵进行初始化。这是**本方案的重要trick之一。** 这个方法能够显著解决某些长尾item的嵌入学习不充分的问题。

```python
# obtain item feat
item_embed_np = np.zeros((item_cnt + 1, 256))
for raw_id, idx in item_raw_id2_idx_dict.items():
    vec = item_content_vec_dict[int(raw_id)]
    item_embed_np[idx, :] = vec
np.save(open(sr_gnn_dir + '/item_embed_mat.npy', 'wb'), item_embed_np)

# initialize node item embedding
if kwargs.get('feature_init', None) is not None:
    init = tf.constant_initializer(np.load(kwargs['feature_init']))
    logger.info("Use Feature Init")
else:
    init = tf.random_uniform_initializer(-self.var_init, self.var_init)

self.node_embedding = (tf.get_variable("node_embedding", shape=[node_count, self.hidden_size], dtype=tf.float32, initializer=init))
```

这里头有一个小细节，点击数据里存在一些特征缺失的items(未出现在item_feat.csv中)，这些items的特征需要做填充。我们采用了局部上下文统计item-item共现关系，并基于共现item的特征做特征填充的方法，这种方式得到的完整item feat对排序过程的提升作用也非常大。

```python
def fill_item_feat(processed_item_feat_df, item_content_vec_dict):
    online_total_click = get_online_whole_click()

    all_click_feat_df = pd.merge(online_total_click, processed_item_feat_df, on='item_id', how='left')

    missed_items = all_click_feat_df[all_click_feat_df['txt_embed_0'].isnull()]['item_id'].unique()
    user_item_time_hist_dict = get_user_item_time_dict(online_total_click)

    # calculate co-occurrence
    co_occur_dict = {}
    window = 5

    def cal_occ(sentence):
        for i, word in enumerate(sentence):
            hist_len = len(sentence)
            co_occur_dict.setdefault(word, {})
            for j in range(max(i - window, 0), min(i + window, hist_len)):
                if j == i or word == sentence[j]: continue
                loc_weight = (0.9 ** abs(i - j))
                co_occur_dict[word].setdefault(sentence[j], 0)
                co_occur_dict[word][sentence[j]] += loc_weight

    for u, hist_item_times in user_item_time_hist_dict.items():
        hist_items = [i for i, t in hist_item_times]
        cal_occ(hist_items)

    # fill
    miss_item_content_vec_dict = {}
    for miss_item in missed_items:
        co_occur_item_dict = co_occur_dict[miss_item]
        weighted_vec = np.zeros(256)
        sum_weight = 0.0
        for co_item, weight in co_occur_item_dict.items():

            if co_item in item_content_vec_dict:
                sum_weight += weight
                co_item_vec = item_content_vec_dict[co_item]
                weighted_vec += weight * co_item_vec

        weighted_vec /= sum_weight
        txt_item_feat_np = weighted_vec[0:128] / np.linalg.norm(weighted_vec[0:128])
        img_item_feat_np = weighted_vec[128:] / np.linalg.norm(weighted_vec[128:])
        cnt_vec = np.concatenate([txt_item_feat_np, img_item_feat_np])
        miss_item_content_vec_dict[miss_item] = cnt_vec

    miss_item_feat_df = pd.DataFrame()
    miss_item_feat_df[item_dense_feat] = pd.DataFrame(miss_item_content_vec_dict.values(),
                                                      columns=item_dense_feat)
    miss_item_feat_df['item_id'] = list(miss_item_content_vec_dict.keys())
    miss_item_feat_df = miss_item_feat_df[['item_id'] + item_dense_feat]

    return miss_item_feat_df, miss_item_content_vec_dict
```



##### 带有节点权重的消息传播

在SR-GNN中，得到物品序列后，将序列中的物品作为图节点，序列中相邻的物品之间通过有向边连接，最终分别得到入边和出边的邻接矩阵并按行归一化。例如，物品序列$s=[v_1, v_2, v_3, v_2, v_4]​$对应的有向图及邻接矩阵$\boldsymbol{M}_{s, out}, \boldsymbol{M}_{s, in}​$,  如下所示:


![img](/picture/machine-learning/sr-gnn-1.png)

![img](/picture/machine-learning/sr-gnn-2.png)

 得到序列的图表示后，之后进行GNN处理的，**遵循GNN信息传递架构 **[12]，即：**信息构造—传播—更新**三个步骤：

1)、 **信息构造：**针对全部物品设置嵌入矩阵，每个节点对应的物品可用嵌入矩阵的一个行向量$\boldsymbol{e}_i \in \mathbb{R}^d$表示。由于训练集中物品呈长尾分布，**对于出现次数较多的物品，我们希望降低它的影响**，因此我们设置节点$i$（即对应的物品$i$）的初始权重，
$$
w_i = \frac{1}{\sqrt{\\#i / \text{median}}+1} \tag{5}
$$

$\\#i$为物品$i$在训练集中出现的次数，$\text{median}$为全部物品出现次数的中位数，最终权重位于(0,1)之间，出现次数较多的物品权重较小，而出现次数较少的物品权重接近1。我们设置权值$w_i$为可学习的参数，因此节点$i$待传播的信息为$w_i \boldsymbol{e_i}$。

2)、 **传播：**按照连接矩阵进行传播，

$$
o_{s,i}^t = \text{concat}(\boldsymbol{M}_{s, in}^i \boldsymbol{E}^{t-1} \boldsymbol{W}_{in}, \boldsymbol{M}_{s, out}^i \boldsymbol{E}^{t-1} \boldsymbol{W}_{out})+b \tag{6}
$$

此处，$\boldsymbol{E}^{t-1}=[w_1 \boldsymbol{e}_1^{t-1}, ..., w_n \boldsymbol{e}_n^{t-1}]$为图中全部节点的信息矩阵，$\boldsymbol{M}_{s, in}, \boldsymbol{M}_{s, out} \in \mathbb{R}^{1 \times n}$分别表示入度矩阵和出度矩阵的$i$行， 我们从入边和出边两个方向传播信息，$o_{s,i}^t \in \mathbb{R}^{2d}$为节点$i$在第$t$步时从邻居节点汇聚得到的信息。$\boldsymbol{W}_{in}, \boldsymbol{W}_{out}, b​$为模型可学习的参数。

3)、 **更新：**根据节点自身的信息和来自邻居的信息，更新节点的信息。这里使用GRU进行结点信息的更新：$\boldsymbol{e_i}^t=GRU(o_{s,i}^t, \boldsymbol{e}_i^{t-1}) + \boldsymbol{e}_i^0​$，此处，我们采用了残差连接。

 以上过程可循环进行$R$步，最终每个节点可获取到它的$R$阶邻居的信息。我们的方案中，$R=2$。

##### **位置编码**

用户的行为受最后一次交互影响较大，为了强化交互顺序的影响，我们增加了位置编码矩阵$P \in \mathbb{R}^{k \times d}$，$k$为位置数量，我们从后向前编码，最后一次交互的物品位置为1，上一次为2，以此类推。通过GNN更新后的节点向量和位置编码向量相加：
$$
\boldsymbol{e_i} \leftarrow \boldsymbol{e}_i + \boldsymbol{p}_i \tag{7}
$$
   $\boldsymbol{p}_i$为节点$i$的位置编码向量。我们设置$k=5$，对于倒数第5个物品之前的物品，它们的位置均为5。

##### **序列级别的嵌入表征**

这里需要汇聚图中全部节点向量，得到一个图级别的输出作为序列的嵌入表征。考虑到最后一次行为的重要性，我们使用了加权平均池化的汇聚方式，即：

$$
\boldsymbol{s}_h = w \boldsymbol{e}_T + (1-w) \sum_{i=1}^{T-1} \frac{\boldsymbol{e_i}}{T-1} \tag{8}
$$

$\boldsymbol{e}_T$为序列最后一个item的嵌入表示，这里我们对序列中最后一个物品之前的物品向量进行平均池化，之后和最后一个物品向量按照权重$w$进行加权，得到序列的表示。$w$是可学习的参数。

##### **预测和损失函数**

   我们对序列向量及物品向量进行L2归一化：
$$
   \boldsymbol{s}_h = \frac{\boldsymbol{s}_h}{||\boldsymbol{s}_h||_2}, \boldsymbol{e}_i = \frac{\boldsymbol{e}_i}{||\boldsymbol{e}_i||_2} \tag{9}
$$
   之后通过点积对物品进行打分：
$$
   \hat{y}_i = \text{softmax}(\sigma \boldsymbol{s}_h^T \boldsymbol{e}_i) \tag{10}
$$
   $\sigma$为超参数，我们设为10，来进一步拉大高意向item和低意向item之间的差距。这实际上是通过余弦相似度对物品进行打分，这些在参考文献[2]中有具体描述。模型的损失为预测概率的多分类交叉熵损失。

### 产出多路召回结果

上述协同过滤方案实际上分为了Item-based，即：item-cf、swing、bi-graph和User-based，即user-cf。在具体进行推荐时，我们封装了基于item的产生召回结果的流程和基于user的产生召回结果的流程。

#### Item-based 

item-based的方法在进行推荐的时候，会利用用户的历史行为item，计算历史行为item最相似的Top-K个item推荐给用户。在计算相似性时，同样会利用前文提到的策略，即：根据交互时间进行指数衰减；根据交互方向进行幂函数衰减。同时，我们还利用了物品的内容特征，即，利用Faiss [11] 计算了item-item之间的内容相似性权重，最后，每个item的得分=召回方法的分数 $\times$ 时间权重 $\times$ 方向权重 $\times$ 内容权重。每种方法产生Top-200个召回结果。

```python
def item_based_recommend(sim_item_corr, user_item_time_dict, user_id, top_k, item_num, alpha=15000,
                         item_cnt_dict=None, user_cnt_dict=None, adjust_type='v2'):
    item_content_sim_dict = get_glv('item_content_sim_dict') # get global variables
    rank = {}
    if user_id not in user_item_time_dict:
        return []
    interacted_item_times = user_item_time_dict[user_id]
    min_time = min([time for item, time in interacted_item_times])
    interacted_items = set([item for item, time in interacted_item_times])

    for loc, (i, time) in enumerate(interacted_item_times):
        if i not in sim_item_corr:
            continue
        for j, wij in sorted(sim_item_corr[i].items(), key=lambda x: x[1], reverse=True)[0:top_k]:
            if j not in interacted_items:
                rank.setdefault(j, 0)

                content_weight = 1.0
                if item_content_sim_dict.get(i, {}).get(j, None) is not None:
                    content_weight += item_content_sim_dict[i][j]
                if item_content_sim_dict.get(j, {}).get(i, None) is not None:
                    content_weight += item_content_sim_dict[j][i]

                time_weight = np.exp(alpha * (time - min_time))
                loc_weight = (0.9 ** (len(interacted_item_times) - loc))
                rank[j] += loc_weight * time_weight * content_weight * wij
  
    if item_cnt_dict is not None:
        for loc, item in enumerate(rank):
            rank[item] = re_rank(rank[item], item, user_id, item_cnt_dict, user_cnt_dict, adjust_type=adjust_type)

    sorted_rank_items = sorted(rank.items(), key=lambda d: d[1], reverse=True)

    return sorted_rank_items[0:item_num]

```

#### User-based

User-based进行推荐时，会将相似用户的历史感兴趣item推荐给目标用户。但是这里面的一个问题是，没有利用到目标用户本身的行为序列信息。我们做了改进，会计算相似用户历史感兴趣item和目标用户本身行为序列中的item之间的相似性，计算相似性时，同样会利用时间权重和方向权重进行衰减。产生Top-200个召回结果。

```python
def user_based_recommend(sim_user_corr, user_item_time_dict, user_id, top_k, item_num, alpha=15000,
                         item_cnt_dict=None, user_cnt_dict=None, adjust_type='v2'):
    item_content_sim_dict = get_glv('item_content_sim_dict')

    rank = {}
    interacted_items = set([i for i, t in user_item_time_dict[user_id]])
    interacted_item_time_list = user_item_time_dict[user_id]
    interacted_num = len(interacted_items)

    min_time = min([t for i, t in interacted_item_time_list])
    time_weight_dict = {i: np.exp(alpha * (t - min_time)) for i, t in interacted_item_time_list}
    loc_weight_dict = {i: 0.9 ** (interacted_num - loc) for loc, (i, t) in enumerate(interacted_item_time_list)}

    for sim_v, wuv in sorted(sim_user_corr[user_id].items(), key=lambda x: x[1], reverse=True)[0:top_k]:
        if sim_v not in user_item_time_dict:
            continue
        for j, j_time in user_item_time_dict[sim_v]:
            if j not in interacted_items:
                rank.setdefault(j, 0)

                content_weight = 1.0
                for loc, (i, t) in enumerate(interacted_item_time_list):
                    loc_weight = loc_weight_dict[i]
                    time_weight = time_weight_dict[i]
                    if item_content_sim_dict.get(i, {}).get(j, None) is not None:
                        content_weight += time_weight * loc_weight * item_content_sim_dict[i][j]

                # weight = np.exp(-15000*abs(j_time-q_time))
                rank[j] += content_weight * wuv

    if item_cnt_dict is not None:
        for loc, item in enumerate(rank):
            rank[item] = re_rank(rank[item], item, user_id, item_cnt_dict, user_cnt_dict, adjust_type=adjust_type)

    rec_items = sorted(rank.items(), key=lambda d: d[1], reverse=True)

    return rec_items[:item_num]
```



#### SR-GNN

我们还对数据进行了增强操作。对每个用户的交互序列进行截断，变成多条的交互序列。然后使用模型进行训练并产出结果。具体使用时，我们使用了两套参数(原始论文实现+改进版实现)训练SR-GNN，每套参数对应的模型根据公式(10)各产生Top-100个召回结果，共Top-200个召回结果。

```python
# Train
python3 {sr_gnn_lib_dir}/main.py --task train --node_count {item_cnt} \
              --checkpoint_path {model_path}/session_id --train_input {file_path}/train_item_seq_enhanced.txt \
              --test_input {file_path}/test_item_seq.txt --gru_step 2 --epochs 10 \
              --lr 0.001 --lr_dc 2 --dc_rate 0.1 --early_stop_epoch 3 --hidden_size 256 --batch_size 256 \
              --max_len 20 --has_uid True --feature_init {file_path}/item_embed_mat.npy --sigma 10 \
              --sq_max_len 5 --node_weight True  --node_weight_trainable True
            
# Output Recommendations          
python3 {sr_gnn_lib_dir}/main.py --task recommend --node_count {item_cnt} --checkpoint_path {checkpoint_path} \
              --item_lookup {file_path}/item_lookup.txt --recommend_output {rec_path} \
              --session_input {file_path}/test_user_sess.txt --gru_step 2 \
              --hidden_size 256 --batch_size 256 --rec_extra_count 50 --has_uid True \
              --feature_init {file_path}/item_embed_mat.npy \
              --max_len 10 --sigma 10 --sq_max_len 5 --node_weight True \
              --node_weight_trainable True
```

在A榜中，单模型的SR-GNN效果已超过4种改进后的CF融合后的效果。

最终，每个用户产生了1000个召回结果。

## 粗排方案

粗排阶段主要基于这样的观察，我们的模型Top 100的hit-rate指标远高于Top 50，说明可能很多低流行度的物品被我们的模型召回了，但是排序较靠面，因此需要提高低频商品的曝光率，以消除对高频商品的偏向性。具体而言，对每个阶段进行召回时，本方案会统计**该阶段内**的物品出现的频次，然后根据该频次以及召回方法计算的item-item相似性分数，对相似性分数进行调整。这是**本方案的key points之一，能够在基本不影响full的情况下，有效提高half**。不同于其他开源的方案，我们在召回后进行re-rank，而不是在精排后进行re-rank。

本方案考虑加入频率因素，具体方法包括：

(1)  首先将频率作为一个单独的考量标准，为了初步鉴定频次的打压效果并尽量排除其他权重对其干扰，直接将召回分数除以物品出现的频次，初步鉴定对half有比较明显的提升，但是会显著降低full指标。

(2)  进一步，经过对item频次进行数据分析，item频率的分布呈现长尾效应，因此对于这些高频但极少数的item，考虑使用幂函数削弱打击的效果。采用的打压函数分段函数如下所示：
$$
\begin{split} 
f(c) &= \log(c + 2), c < 4 \\
f(c) &= c, \ 4 \leq  c < 10 \\
f(c) &=  c^{0.75} + 5.0, c > 10
\end{split}
$$
其中，$c$为item出现在目标pahse中的频次，则新的分数为，$s_i^{*} = s_i  /  f(c_i)$。

(3) 考虑到不同活跃度的用户对于不同频率的物品的倾向性不同，越活跃的用户越倾向于点击低频的商品，因此对高活跃度的用户，需要提高高频item打压程度；对低活跃度的用户，提高对于低频率物品的打压程度。对于不同用户进行区分的策略在几乎不影响ndcg-full同时，有效提高了ndcg-half。

```python
def re_rank(sim, i, u, item_cnt_dict, user_cnt_dict, adjust_type='v2'):
    '''
    :param sim: recall sim value
    :param i: item
    :param u: user
    :param item_cnt_dict: item frequency map
    :param user_cnt_dict: user frequency map
    :param adjust_type: re-rank strategy, v0, v1, v2
    :return:
    '''
    if adjust_type is None:
        return sim
    elif adjust_type == 'v1':
        # Log，Linear, 3/4, only consider item frequency
        if item_cnt_dict.get(i, 1.0) < 4:
            heat = np.log(item_cnt_dict.get(i, 1.0) + 2)
        elif item_cnt_dict.get(i, 1.0) >= 4 and item_cnt_dict.get(i, 1.0) < 10:
            heat = item_cnt_dict.get(i, 1.0)
        else:
            heat = item_cnt_dict.get(i, 1.0) ** 0.75 + 5.0  # 3/4
        sim *= 2.0 / heat

    elif adjust_type == 'v2':
        # Log，Linear, 3/4, consider user activity
        user_cnt = user_cnt_dict.get(u, 1.0)

        if item_cnt_dict.get(i, 1.0) < 4:
            heat = np.log(item_cnt_dict.get(i, 1.0) + 2)
        # 对低活跃度的用户，提高对于低频率物品的打压程度
        elif item_cnt_dict.get(i, 1.0) >= 4 and item_cnt_dict.get(i, 1.0) < 10:
            if user_cnt > 50:
                heat = item_cnt_dict.get(i, 1.0) * 1
            elif user_cnt > 25:
                heat = item_cnt_dict.get(i, 1.0) * 1.2
            else:
                heat = item_cnt_dict.get(i, 1.0) * 1.6
        # 对高活跃度的用户，需要提高高频item打压程度
        else:
            if user_cnt > 50:
                user_cnt_k = 0.4
            elif user_cnt > 10:
                user_cnt_k = 0.1
            else:
                user_cnt_k = 0
            heat = item_cnt_dict.get(i, 1.0) ** user_cnt_k + 10 - 10 ** user_cnt_k
        sim *= 2.0 / heat
    else:
        sim += 2.0 / item_cnt_dict.get(i, 1.0)
    return sim
```

不同的召回方法得到的分数会经过上述步骤进行分数调整粗排，然后需要将不同召回模型初步融合在一起，我们的方法是，每种方法对**每个用户**产生的推荐结果先进行打分的最小最大归一化；然后求和合并不同方法对同一个用户的打分结果。

**注**：实际上，我们临近截止日期的时候对召回做了小修改，full指标上升了一些，half下降了一些，导致覆盖了原本最好的half结果，没来的及对改进后的召回重排策略进行精排。**最终导致目前线上最终的成绩是仅通过上述召回方案得到的**。而在我们所有的提交记录中，我们最好的half指标的成绩是该召回方案和下文即将描述的排序方案产生的。笔者认为，如果对改进后的重排策略进行精排的话，我们的分数应该还会更高。

![results](/picture/machine-learning/debiasing_results.png)

## 精排方案

到目前为止，B榜的最终成绩(full rank 3rd, half rank 10th)仅由上文提到的召回+粗排即可得到。精排方案在A榜的时候会有full会有0.05+的提升；half会有0.01+的提升。B榜由于时间问题没来得及对改进后的召回方案做排序并提交。如果你有兴趣可以接着往下阅读。

精排方案主要由GBDT和DIN方法组成。这里面最重要的步骤来自于训练样本的构造和特征的构造。其中，训练样本的构造是重中之重，个人认为也是本次比赛**排序阶段最大的难点所在**。

### 训练样本构造

排序方案的训练样本构造我们采用了序列推荐的典型构造方案，即：滑窗方式构造训练样本。为了保证训练时和线上预测时的数据一致性，我们以行为序列中的1个item为滑窗步长，共滑动了10步。具体步骤即：对每个用户的行为序列$s=[i_1, i_2, ..., i_n]$，从倒数第1个item开始，即：$i_n$为ground truth item, $i_1 \sim i_{n-1}$的通过我们的召回模型来计算item pair相似性，并为第$n$个位置的next-item产生召回结果；滑动窗口往左滑动1步，即：$i_{n-1}$为ground truth item, $i_1 \sim i_{n-2}$的通过我们的召回模型来计算item pair相似性，并为第$n-1$个位置的next-item产生召回结果；以此类推，共滑动10步。这种方式的缺点在于，计算复杂度非常高。因为每次滑动，都需要进行相似性的计算，并用训练集中所有的用户进行召回。目前笔者还不清楚这种方式是否是最优的构造方法(应该不是最优的)，希望后面看看其他队伍的开源开案，学习学习。

```python
def sliding_obtain_training_df(c, is_silding_compute_sim=False):
    print('train_path={}, test_path={}'.format(train_path, test_path))

    all_click, click_q_time = get_phase_click(c)

    # for validation
    compute_mode = 'once' if not is_silding_compute_sim else 'multi'

    save_training_path = os.path.join(user_data_dir, 'training', mode, compute_mode, str(c))
    click_history_df = all_click
    recall_methods = {'item-cf', 'bi-graph', 'user-cf', 'swing'}

    if not os.path.exists(save_training_path): os.makedirs(save_training_path)

    total_step = 10
    step = 0
    full_sim_pair_dict = get_multi_source_sim_dict_results_multi_processing(click_history_df,
                                                                            recall_methods=recall_methods)
    pickle.dump(full_sim_pair_dict, open(os.path.join(save_training_path, 'full_sim_pair_dict.pkl'), 'wb'))

    step_user_recall_item_dict = {}
    step_strategy_sim_pair_dict = {}

    while step < total_step:
        print('step={}'.format(step))
        click_history_df, click_last_df = get_history_and_last_click_df(click_history_df)  # override click_history_df
        user_item_time_dict = get_user_item_time_dict(click_history_df)

        if is_silding_compute_sim:
            sim_pair_dict = get_multi_source_sim_dict_results_multi_processing(click_history_df,
                                                                               recall_methods=recall_methods)  # re-compute
        else:
            sim_pair_dict = full_sim_pair_dict

        user_recall_item_dict = do_multi_recall_results_multi_processing(sim_pair_dict, user_item_time_dict,
                                                                         ret_type='tuple',
                                                                         recall_methods=recall_methods)

        step_user_recall_item_dict[step] = user_recall_item_dict
        if is_silding_compute_sim:
            step_strategy_sim_pair_dict[step] = sim_pair_dict
        step += 1

    pickle.dump(step_user_recall_item_dict,
                open(os.path.join(save_training_path, 'step_user_recall_item_dict.pkl'), 'wb'))

    if is_silding_compute_sim:
        pickle.dump(step_strategy_sim_pair_dict,
                    open(os.path.join(save_training_path, 'step_strategy_sim_pair_dict.pkl'), 'wb'))

    # validation/test recall results based on full_sim_pair_dict
    # user-cf depend on sim-user history, so use all-click; test user history will not occur in train, so it's ok
    print('obtain validate/test recall data')
    if mode == 'offline':
        all_user_item_dict = get_user_item_time_dict(all_click)

        val_user_recall_item_dict = do_multi_recall_results_multi_processing(full_sim_pair_dict,
                                                                             all_user_item_dict,
                                                                             target_user_ids=click_q_time['user_id'].unique(), ret_type='tuple',
                                                                             recall_methods=recall_methods)
        pickle.dump(val_user_recall_item_dict,
                    open(os.path.join(save_training_path, 'val_user_recall_item_dict.pkl'), 'wb'))
```



构造样本标签时，将召回结果中，命中了的用户真实点击的item作为正样本（即：不包括召回结果中未命中，但是用户真实点击的item，好处是能够把**召回分数特征等**送到模型中进行排序），然后随机负采样部分item作为负样本，负样本的策略以user和item侧分别入手，按照比例进行负采样，最终采样到的负样本: 正样本比例 约等于 10:1。

具体实现时，我们会对每个阶段的数据中的所有用户，分别进行召回并构造样本和标签。上述得到的数据格式即：user id, item id, hist item sequence, label，即: 用户id，目标物品id，用户历史交互item序列，标签。

### 特征提取

重要的特征主要涉及**召回时的特征**以及**目标item和用户历史item之间的各种关联性**，如内容关联性、行为关联性等。

#### 召回特征

召回特征主要包括了：

- 用户对目标item的分数，即多种recall方法融合并粗排后的分数； 
- 目标item和历史item的相似性，我们只选择历史交互序列中的**最后3个物品**的内容特征进行计算相似性。

#### 内容特征

- 待预测的目标物品原始的内容特征。

- 用户历史交互序列中的item的内容特征，根据交互位置顺序进行加权计算后的兴趣向量。
- 用户兴趣向量和物品内容向量之间的内容相似性分数。
- word2vec对训练集中的用户hist item sequences进行训练，然后得到的每个物品的w2v向量。
- 每个待预测的目标物品的w2v向量和用户历史交互的item的w2v向量之间的相似性分数。

#### ID特征

这部分特征主要供深度学习模型DIN使用。包括：

- user id特征
- item id特征
- 用户历史行为序列中的 item id特征 （和item id 特征共享嵌入）

比较遗憾的是，本次比赛user侧的特征由于缺失值过多，我们没有花太多时间在user侧的特征提取，比如像item侧的特征一样，进行缺失值预测、补全等。

### 排序模型

排序模型包括了两个，1个是GBDT，这里我们采用了LightGBM [5] 中的learning to rank方法LGBMRanker进行排序。另一个是DIN，这里采用了DeepCTR [6] 库中的DIN实现版本。对于DIN，我们利用了物品的内容特征对item的嵌入进行了初始化；利用用户历史行为序列中的item的加权后的兴趣向量对user的嵌入进行了初始化。

- GBDT实现：

```python
def lgb_main(train_final_df, val_final_df=None):
    print('ranker begin....')
    train_final_df.sort_values(by=['user_id'], inplace=True)
    g_train = train_final_df.groupby(['user_id'], as_index=False).count()["label"].values

    if mode == 'offline':
        val_final_df = val_final_df.sort_values(by=['user_id'])
        g_val = val_final_df.groupby(['user_id'], as_index=False).count()["label"].values

    lgb_ranker = lgb.LGBMRanker(
        boosting_type='gbdt', num_leaves=31, reg_alpha=0.0, reg_lambda=1,
        max_depth=-1, n_estimators=300, objective='lambdarank',
        subsample=0.7, colsample_bytree=0.7, subsample_freq=1,
        learning_rate=0.01, min_child_weight=50, random_state=2018,
        n_jobs=-1)  # 300epoch, best, 0.882898, dense_feat  + hist_cnt_sim_feat user_interest_dense_feat

    if mode == 'offline':
        lgb_ranker.fit(train_final_df[lgb_cols], train_final_df['label'], group=g_train,
                       eval_set=[(val_final_df[lgb_cols], val_final_df['label'])], eval_group=[g_val],
                       eval_at=[50], eval_metric=['auc', ],
                       early_stopping_rounds=50, )
    else:
        lgb_ranker.fit(train_final_df[lgb_cols], train_final_df['label'], group=g_train)

    print('train done...')
    return lgb_ranker
```

- DIN实现：

```python
def din_main(target_phase, train_final_df, val_final_df=None):
    print('din begin...')
    get_init_user_embed(target_phase, is_use_whole_click=True)
    feature_names, linear_feature_columns, dnn_feature_columns = generate_din_feature_columns(train_final_df,
                                                                                              ['user_id',
                                                                                               'item_id'],
                                                                                              dense_features=item_dense_feat + sim_dense_feat + hist_time_diff_feat + hist_cnt_sim_feat + user_interest_dense_feat)
    train_input = {name: np.array(train_final_df[name].values.tolist()) for name in feature_names}
    train_label = train_final_df['label'].values
    if mode == 'offline':
        val_input = {name: np.array(val_final_df[name].values.tolist()) for name in feature_names}
        val_label = val_final_df['label'].values

    EPOCH = 1
    behavior_feature_list = ['item_id']
    model = KDD_DIN(dnn_feature_columns, behavior_feature_list, dnn_hidden_units=HIDDEN_SIZE,
                    att_hidden_size=(128, 64), att_weight_normalization=True,
                    dnn_dropout=0.5)

    model.compile(optimizer=tf.keras.optimizers.Adam(lr=3e-4), loss="binary_crossentropy",
                  metrics=['binary_crossentropy', tf.keras.metrics.AUC()], )

    if mode == 'offline':
        model.fit(train_input, train_label, batch_size=BATCH_SIZE, epochs=EPOCH,
                  verbose=1, validation_data=(val_input, val_label), ) 
    else:
        model.fit(train_input, train_label, batch_size=BATCH_SIZE, epochs=EPOCH,
                  verbose=1)
    return model, feature_names

```



### 模型集成

最后，我们将GBDT预测的分数和DIN预测的分数融合起来。具体而言，每个方法的预测概率会先进行user-wise的归一化操作；然后两个方法归一化后预测的概率值进行相加融合。最后按照融合后的分数进行排序，并产生最终的Top 50结果。在A榜的时候，lgb对召回结果对full指标的提升效果大概都在0.02+；但是融合后的LGB+DIN，提升效果可达到0.05+。对half指标的提升略微少了一些，可能原因在于模型过于关注召回得到的sim等特征，对debiasing相关的特征挖掘比较少。

```python
def norm_sim(sim_df, weight=0.0):
    # print(sim_df.head())
    min_sim = sim_df.min()
    max_sim = sim_df.max()
    if max_sim == min_sim:
        sim_df = sim_df.apply(lambda sim: 1.0)
    else:
        sim_df = sim_df.apply(lambda sim: 1.0 * (sim - min_sim) / (max_sim - min_sim))

    sim_df = sim_df.apply(lambda sim: sim + weight)  # plus one
    return sim_df


def ensemble(output_ranking_filename):
    # ensemble lgb+din
    lgb_output_file = 'ranker-' + output_ranking_filename + '-pkl'
    # read lgb
    lgb_ranker_df = pickle.load(
        open('{}/{}'.format(output_path, lgb_output_file), 'rb'))
    lgb_ranker_df['sim'] = lgb_ranker_df.groupby('user_id')['sim'].transform(lambda df: norm_sim(df))

    # read din
    din_output_file = 'din-' + output_ranking_filename + '-pkl'
    din_df = pickle.load(
        open('{}/{}'.format(output_path, din_output_file), 'rb'))
    din_df['sim'] = din_df.groupby('user_id')['sim'].transform(lambda df: norm_sim(df))

    # fuse lgb and din
    din_lgb_full_df = lgb_ranker_df.append(din_df)
    din_lgb_full_df = din_lgb_full_df.groupby(['user_id', 'item_id', 'phase'])['sim'].sum().reset_index()

    online_top50_click_np, online_top50_click = obtain_top_k_click()
    res3 = get_predict(din_lgb_full_df, 'sim', online_top50_click)
    res3.to_csv(output_path + '/result.csv', index=False, header=None)
```



# 总结

对本文方案的**key points**作一个总结：

- **召回训练集的构造**，如何使用全量数据进行训练，user侧和item侧都需要**防止穿越**。这个提高非常显著，说明**数据**对于结果的影响非常大。
- **CF中的改进点能够有效进行纠偏**，包括，**交互时间、方向、内容相似性、物品流行度、用户活跃度**等。这个提高也很显著，和赛题Debiasing主题契合。
- **SR-GNN**基于序列推荐的图神经网络模型，完美契合本次比赛序列推荐场景，捕捉item之间的**多阶相似性**并兼顾用户**长短期偏好**。另外，我们基于SR-GNN的改进点，**使用内容特征进行嵌入初始化**、根据频次引入结点权重 (为了纠偏)、位置编码 (强化短期交互互影响力)、嵌入归一化、残差连接、sequence-level embedding的构建等都带来了提升。SR-GNN召回方案的提升效果达到**0.05+**。
- **粗排考虑了频次，提高低频商品的曝光率，以消除召回方法对高频商品的偏向性**，对half指标的提升很显著。
- **排序特征的构建，**包括召回特征、内容特征、历史行为相关的特征、ID特征等。
- **排序模型集成**，**LGB和DIN模型的融合**，对最终的指标有比较高的提升。

# 参考文献

[1]  Wu S, Tang Y, Zhu Y, et al. Session-based recommendation with graph neural networks[C]//Proceedings of the AAAI Conference on Artificial Intelligence. 2019, 33: 346-353.

[2]  Gupta P, Garg D, Malhotra P, et al. NISER: Normalized Item and Session Representations with Graph Neural Networks[J]. arXiv preprint arXiv:1909.04276, 2019.

[3]  Zhou T, Ren J, Medo M, et al. Bipartite network projection and personal recommendation[J]. Physical review E, 2007, 76(4): 046115.

[4] Zhou G, Zhu X, Song C, et al. Deep interest network for click-through rate prediction[C]//Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2018: 1059-1068.

[5] Ke G, Meng Q, Finley T, et al. Lightgbm: A highly efficient gradient boosting decision tree[C]//Advances in neural information processing systems. 2017: 3146-3154.

[6] DeepCTR, Easy-to-use,Modular and Extendible package of deep-learning based CTR models, https://github.com/shenweichen/DeepCTR

[7] A simple itemCF Baseline, score:0.1169, https://tianchi.aliyun.com/forum/postDetail?postId=103530

[8] 改进青禹小生baseline，phase3线上0.2, https://tianchi.aliyun.com/forum/postDetail?postId=105787

[9] 推荐系统算法调研, http://xtf615.com/2018/05/03/recommender-system-survey/

[10] A Simple Recall Method based on Network-based Inference, score:0.18 (phase0-3), https://tianchi.aliyun.com/forum/postDetail?postId=104936

[11] A library for efficient similarity search and clustering of dense vectors, https://github.com/facebookresearch/faiss

[12] CIKM 2019 tutorial: Learning and Reasoning on Graph for Recommendation, https://next-nus.github.io/

[13] Source code and datasets for the paper "Session-based Recommendation with Graph Neural Networks" (AAAI-19), https://github.com/CRIPAC-DIG/SR-GNN

也欢迎关注我的公众号"**蘑菇先生学习记**"，更快更及时地获取推荐系统前沿进展！

![qr](/picture/qr_sr_code.png)