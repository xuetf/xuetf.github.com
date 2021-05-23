---
title: Transformers源码阅读和实践
date: 2020-07-05 22:18:15
tags: [nlp,bert,深度学习,预训练模型]
comments: true
categories: 自然语言处理
top: 21
---

本文主要针对HuggingFace开源的 [transformers](https://github.com/huggingface/transformers)，以BERT为例介绍其源码并进行一些实践。主要以pytorch为例 (tf 2.0 代码风格几乎和pytorch一致)，介绍BERT使用的Transformer Encoder，Pre-training Tasks和Fine-tuning Tasks。最后，针对预训练好的BERT进行简单的实践，例如产出语句embeddings，预测目标词以及进行抽取式问答。本文主要面向BERT新手，在阅读本文章前，假设读者已经阅读过[BERT](https://arxiv.org/abs/1810.04805)原论文。

<!--more-->

## Core Components

[Transformers: State-of-the-art Natural Language Processing](https://arxiv.org/pdf/1910.03771.pdf)

参考上面的论文，transformers开源库的核心组件包括3个：

- **Conﬁguration**：配置类，通常继承自**PretrainedConﬁg**，保存model或tokenizer的超参数，例如词典大小，隐层维度数，dropout rate等。配置类主要可用于复现模型。
- **Tokenizer**：切词类，通常继承自**PreTrainedTokenizer**，主要存储词典，token到index映射关系等。此外，还会有一些model-specific的特性，如特殊token，[SEP], [CLS]等的处理，token的type类型处理，语句最大长度等，因此tokenizer通常和模型是一对一适配的。比如BERT模型有BertTokenizer。Tokenizer的实现方式有多种，如word-level, character-level或者subword-level，其中subword-level包括[Byte-Pair-Encoding](https://arxiv.org/abs/1508.07909)，[WordPiece](https://research.google/pubs/pub37842/)。subword-level的方法目前是transformer-based models的主流方法，能够有效解决OOV问题，学习词缀之间的关系等。Tokenizer主要为了**将原始的语料编码成适配模型的输入。**
- **Model**: 模型类。封装了预训练模型的计算图过程，遵循着相同的范式，如根据token ids进行embedding matrix映射，紧接着多个self-attention层做编码，最后一层task-specific做预测。除此之外，Model还可以做一些灵活的扩展，用于下游任务，例如在预训练好的Base模型基础上，添加task-specific heads。比如，language model heads，sequence classiﬁcation heads等。在代码库中通常命名为，**XXXForSequenceClassification** or **XXXForMaskedLM**，其中XXX是模型的名称（如Bert）， 结尾是预训练任务的名称 (MaskedLM) 或下游任务的类型(SequenceClassification)。

另外，针对上述三大类，transformer还额外封装了**AutoConfig, AutoTokenizer,AutoModel**，可通过模型的命名来定位其所属的具体类，比如'bert-base-cased'，就可以知道要加载BERT模型相关的配置、切词器和模型。非常方便。通常上手时，我们都会用Auto封装类来加载切词器和模型。

## Transformer-based Pre-trained model

所有已实现的Transformer-based Pre-trained models: 

```python
CONFIG_MAPPING = OrderedDict(
    [
        ("retribert", RetriBertConfig,),
        ("t5", T5Config,),
        ("mobilebert", MobileBertConfig,),
        ("distilbert", DistilBertConfig,),
        ("albert", AlbertConfig,),
        ("camembert", CamembertConfig,),
        ("xlm-roberta", XLMRobertaConfig,),
        ("marian", MarianConfig,),
        ("mbart", MBartConfig,),
        ("bart", BartConfig,),
        ("reformer", ReformerConfig,),
        ("longformer", LongformerConfig,),
        ("roberta", RobertaConfig,),
        ("flaubert", FlaubertConfig,),
        ("bert", BertConfig,),
        ("openai-gpt", OpenAIGPTConfig,),
        ("gpt2", GPT2Config,),
        ("transfo-xl", TransfoXLConfig,),
        ("xlnet", XLNetConfig,),
        ("xlm", XLMConfig,),
        ("ctrl", CTRLConfig,),
        ("electra", ElectraConfig,),
        ("encoder-decoder", EncoderDecoderConfig,),
    ]
```

上述是该开源库实现的模型，包括了BERT，GPT2，XLNet，RoBERTa，ALBERT，ELECTRA，T5等家喻户晓的预训练语言模型。

下面将以BERT为例，来介绍BERT相关的源码。建议仔细阅读源码中我做的一些**注释**，尤其是**步骤的细分**。同时，关注下目录的层次，**即：不同类之间的关系。**

## BertModel Transformer

**BertModel**, The bare Bert Model transformer outputting **raw hidden-states** without any specific head on top。这个类的目标主要就是利用**Transformer**获取序列的编码向量。抽象出来的目标是为了适配不同的预训练任务。例如：MLM预训练任务对应的类为BertForMaskedLM，其中有个成员实例为BertModel，就是为了编码序列，获取序列的hidden states后，再构建MaskedLM task进行训练或者预测。

核心构造函数和Forward流程代码如下：

```python
# BertModel的构造函数
def __init__(self, config):
    super().__init__(config)
    self.config = config
    self.embeddings = BertEmbeddings(config)
    self.encoder = BertEncoder(config)
    self.pooler = BertPooler(config)
    self.init_weights()
    
def forward(self, input_ids=None, attention_mask=None,token_type_ids=None,    
            position_ids=None, head_mask=None, inputs_embeds=None,
            encoder_hidden_states=None, encoder_attention_mask=None,
            output_attentions=None, output_hidden_states=None,):
    # ignore some code here...
    
    # step 1: obtain sequence embedding, BertEmbeddings 
    embedding_output = self.embeddings(
        input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, 
        inputs_embeds=inputs_embeds)
    
    # step 2: transformer encoder, BertEncoder
    encoder_outputs = self.encoder(
        embedding_output,
        attention_mask=extended_attention_mask,
        head_mask=head_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_extended_attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
    )
    sequence_output = encoder_outputs[0]
    
    # step 3: pooling to obtain sequence-level encoding, BertPooler
    pooled_output = self.pooler(sequence_output)

    outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
    return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)
```

**参数如下：**

- **input_ids**: 带特殊标记([CLS]、[SEP])的**token ids**序列, e.g., ``tensor([[ 101, 1188, 1110, 1126, 7758, 1859,  102]])``, 其中101和102分别是[CLS]，[SEP]对应的token id。 其**shape**: $B \times S$，**B**为batch size, **S**为序列的长度，此例即：1x7。
- **inputs_embeds:** 和input_ids参数**二选一**。inputs_embeds代表给定了输入tokens对应的token embeddings，比如用word2vec的word embeddings作为token embeddings，这样就不需要用input_ids对默认随机初始化的embedding做lookup得到token embeddings。
- **attention_mask**: **self-attention使用**，可选，shape和input_ids一致。当对encoder端的序列做self-attention时，默认全为1，即都可以attend；decoder端序列做self-attention时，默认为类似下三角矩阵的形式 (对角线也为1)。
- **token_type_ids**: 可选，shape和input_ids一致，单语句输入时，取值全为0；在“语句对“的输入中，该取值为0或1，即：前一句为0，后一句为1。
- **head_mask**: **self-attention使用，**可选，想用哪些head，就为1或者None，不想用的head就为0。shape为[num_heads] or [num_hidden_layers x num_heads]，即：可以每层每个head单独设置mask。
- **position_ids**: 可选，位置id，默认就是0~S。
- **encoder_hidden_states/encoder_attention_mask**：decoder端对encoder端做cross-attention时使用，此时K和V即通过encoder_hidden_states得到。

其中，

- **Step 1**: **获取序列的embedding**，对应下文要介绍的**BertEmbeddings**
- **Step 2**: **利用Transformer进行编码**，对应下文要介绍的**BertEncoder**，获取sequence token-level encoding.
- **Step 3**: **对 [CLS] 对应的hidden state进行非线性变换得到** sequence-level encoding，对应下文要介绍的**BertPooler**。

### BertEmbeddings

**第一步Step 1**，获取序列的embeddings

**token embedding + position embedding + segment embedding**

```python
embedding_output = self.embeddings(
    input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds) # embeddings是BertEmbeddings类
```

- 基于input_ids或者inputs_embeds获取token embeddings。
- 基于position_ids获取position embeddings，此处采用的是绝对位置编码。
- 基于token_type_ids获取语句的segment embeddings。

```python
# BertEmbeddings core forward code: 
def forward(self, input_ids=None, token_type_ids=None,
            position_ids=None, inputs_embeds=None):
    # ignore some codes here...
    # step 1: token embeddings
    if inputs_embeds is None:
        inputs_embeds = self.word_embeddings(input_ids) # token embeddings
    # step 2: position embeddings
    position_embeddings = self.position_embeddings(position_ids)
    # step 3: segment embeddings
    token_type_embeddings = self.token_type_embeddings(token_type_ids)

    embeddings = inputs_embeds + position_embeddings + token_type_embeddings
    embeddings = self.LayerNorm(embeddings)
    embeddings = self.dropout(embeddings)
    return embeddings
```

此处还做了layer_norm和dropout。输出的embedding的shape为，$B \times S \times D$。D默认为768。此处输出的embeddings标记为$X$。

### BertEncoder

**第二步，step 2**，利用**Transformer**对序列进行编码

```python
# encoder是BertEncoder类
encoder_outputs = self.encoder(
     embedding_output, # 序列embedding, B x S x D
     attention_mask=extended_attention_mask, # 序列self-attention时使用
     head_mask=head_mask, # 序列self-attention时使用
     encoder_hidden_states=encoder_hidden_states, # decoder，cross-attention
     encoder_attention_mask=encoder_extended_attention_mask, # cross-attention
     output_attentions=output_attentions, # 是否输出attention
     output_hidden_states=output_hidden_states)  # 是否输出每层的hidden state
```

- **embedding_output**：BertEmbeddings的输出，batch中样本序列的每个token的嵌入。$B \times S \times D$
- **extended_attention_mask**：**self-attention**使用。根据attention_mask做维度广播$(B \times H \times  S \times S)$，$H$是head数量，此时，方便下文做self-attention时作mask，即：softmax前对logits作处理，**logits+extended_attention_mask**，即：attention_mask取值为1时，extended_attention_mask对应位置的取值为0；否则，attention_mask为0时，extended_attention_mask对应位置的取值为-10000.0 (很小的一个数)，这样softmax后，mask很小的值对应的位置概率接近0达到mask的目的。
- **head_mask**：**self-attention**使用。同样可以基于**原始输入head_mask作维度广播**，广播前的shape为H or L x H；广播后的shape为：**L x B x H x S x S**。即每个样本序列中每个token对其他tokens的head attentions 值作mask，head attentions数量为L x H。
- **encoder_hidden_states**：可选，**cross-attention使用**。即：decoder端做编码时，要传入encoder的隐状态，**B x S x D**。
- **encoder_attention_mask**：可选，**cross-attention使用**。即，decoder端做编码时，encoder的隐状态的attention mask。和extended_attention_mask类似，**B x S。**
- **output_attentions**：是否输出attention值，bool。可用于可视化attention scores。
- **output_hidden_states**：是否输出每层得到的隐向量，bool。

```python
# BertEncoder由12层BertLayer构成
self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
```

```python
# BertEncoder Forward核心代码
def forward(self, hidden_states,
        attention_mask=None, head_mask=None,
        encoder_hidden_states=None, encoder_attention_mask=None,
        output_attentions=False, output_hidden_states=False):
    # ignore some codes here...
    all_hidden_states = ()
    all_attentions = ()
    for i, layer_module in enumerate(self.layer): # 12层BertLayer
        if output_hidden_states:
           all_hidden_states = all_hidden_states + (hidden_states,)
		   # step 1: BertLayer iteration
           layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                head_mask[i],
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions) # BertLayer Forward，核心！！！

           hidden_states = layer_outputs[0] # overide for next iteration

           if output_attentions:
               all_attentions = all_attentions + (layer_outputs[1],) # 存每层的attentions，可以用于可视化

    # Add last layer
    if output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    outputs = (hidden_states,)

    if output_hidden_states:
       outputs = outputs + (all_hidden_states,)

    if output_attentions:
       outputs = outputs + (all_attentions,)

    return outputs  # last-layer hidden state, (all hidden states), (all attentions)
```

#### BertLayer

上述代码最重要的是循环内的**BertLayer**迭代过程，其核心代码：

```python
def forward(self, hidden_states, attention_mask=None, head_mask=None,
            encoder_hidden_states=None, encoder_attention_mask=None,
            output_attentions=False,):
    # step 1.0: self-attention, attention实例是BertAttention类
    self_attention_outputs = self.attention(
        hidden_states, attention_mask, head_mask, output_attentions=output_attentions,
    )
    attention_output = self_attention_outputs[0]
    outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

    # step 1.1: 如果是decoder, 就作cross-attention，此时step1.0的输出即为decoder侧的序列的self-attention结果，并作为step1.1的输入；step 1.1的输出为decoder侧的cross-attention结果, crossattention实例也是BertAttention
    if self.is_decoder and encoder_hidden_states is not None:
        cross_attention_outputs = self.crossattention(
            attention_output,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
        )
        attention_output = cross_attention_outputs[0]
        outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

    # step 2: intermediate转化，对应原论文中的前馈神经网络FFN
    intermediate_output = self.intermediate(attention_output)
    # step 3: 做skip-connection
    layer_output = self.output(intermediate_output, attention_output)
    outputs = (layer_output,) + outputs
    return outputs
```

其中，step 1分为了2个小步骤。如果是encoder (BERT只用了encoder)，只有1.0起作用，即只对输入序列进行self-attention。如果是做seq2seq的模型，还会用到transformer的decoder，此时1.0就是对decoder的seq做self-attention，相应的attention_mask实际上是类下三角形式的矩阵；而1.1步骤此时就是基于1.0得到的self-attention序列的hidden states，对encoder_hidden_states进行cross-attention。这是本部分的重点。

##### BertAttention

BertAttention是上述代码中attention实例对应的类，也是transformer进行self-attention的核心类。包括了BertSelfAttention和BertSelfOutput成员。

```python
class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        
    def forward(self, hidden_states, attention_mask=None,
        		head_mask=None, encoder_hidden_states=None,
        		encoder_attention_mask=None, output_attentions=False):
        
        # step 1: self-attention, B x S x D
        self_outputs = self.self(
            hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, output_attentions)
        
        # step 2: skip-connection, B x S x D
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs
```

- **BertSelfAttention**: 是**self-attention**，BertSelfAttention可以被实例化为encoder侧的self-attention，也可以被实例化为decoder侧的self-attention，此时attention_mask是非空的 (类似下三角形式的mask矩阵)。同时，还可以实例化为decoder侧的cross-attention，此时，hidden_states即为decoder侧序列的self-attention结果，同时需要传入encoder侧的encoder_hidden_states和encoder_attention_mask来进行cross-attention。

  ```python
  def forward(self, hidden_states, attention_mask=None, head_mask=None,
      		encoder_hidden_states=None, encoder_attention_mask=None,
      		output_attentions=False):
      # step 1: mapping Query/Key/Value to sub-space
      # step 1.1: query mapping
      mixed_query_layer = self.query(hidden_states) # B x S x (H*d)
      
      # If this is instantiated as a cross-attention module, the keys
      # and values come from an encoder; the attention mask needs to be
      # such that the encoder's padding tokens are not attended to.
      
      # step 1.2: key/value mapping
      if encoder_hidden_states is not None:
          mixed_key_layer = self.key(encoder_hidden_states) # B x S x (H*d)
          mixed_value_layer = self.value(encoder_hidden_states) 
          attention_mask = encoder_attention_mask 
      else:
          mixed_key_layer = self.key(hidden_states) # B x S x (H*d)
          mixed_value_layer = self.value(hidden_states)
  
      query_layer = self.transpose_for_scores(mixed_query_layer) # B x H x S x d
      key_layer = self.transpose_for_scores(mixed_key_layer) # B x H x S x d
      value_layer = self.transpose_for_scores(mixed_value_layer) # B x H x S x d
  
      # step 2: compute attention scores
      
      # step 2.1: raw attention scores
      # B x H x S x d   B x H x d x S -> B x H x S x S
      # Take the dot product between "query" and "key" to get the raw attention scores.
      attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
      attention_scores = attention_scores / math.sqrt(self.attention_head_size)
      
      # step 2.2: mask if necessary
      if attention_mask is not None:
         # Apply the attention mask, B x H x S x S
      	attention_scores = attention_scores + attention_mask
  
      # step 2.3: Normalize the attention scores to probabilities, B x H x S x S
      attention_probs = nn.Softmax(dim=-1)(attention_scores)
  
      # This is actually dropping out entire tokens to attend to, which might
      # seem a bit unusual, but is taken from the original Transformer paper.
      attention_probs = self.dropout(attention_probs)
  
      # Mask heads if we want to
      if head_mask is not None:
          attention_probs = attention_probs * head_mask
  	# B x H x S x S   B x H x S x d ->  B x H x S x d
      
      # step 4: aggregate values by attention probs to form context encodings
      context_layer = torch.matmul(attention_probs, value_layer)
  	# B x S x H x d
      context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
      # B x S x D
      new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
      # B x S x D，相当于是多头concat操作
      context_layer = context_layer.view(*new_context_layer_shape)
  
      outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
      return outputs
  ```
  不同head均分768维度，12个head则每个为64维度；具体计算的时候合在一起，即同时算multi-head。记本步骤的输出为：$\text{Multi-head}(X)$ ，输入$X$即为hidden_states参数。

  - $Q、K、V$的shape:  $B \times S \times (H*d)$ : **<Batch Size, Seq Length, Head Num, Embedding Dimension>**，$D=H*d$。此处D=768, H=12, d=64。
  - **attention score计算过程:** 
    - $Q,K,V$:  $B \times H \times S \times d $，**transpose_for_scores**。
    - $K^T$:  $B \times H \times d \times S$
    - $QK^T$, $B \times H \times S \times S$
    - $\text{logit}=QK^T/\sqrt{D}$, 如果是decoder侧的self-attention，则logit加上预先计算好的decoder侧对应的序列的每个位置的attention_mask，实际上就是下三角形式(包括对角线)的mask矩阵。
    - $p=\text{softmax}(\text{logit})$, $B \times H \times S \times S$：每个batch每个head内，每个token对序列内其它token的attention score。
  - **context_layer**:  $p*V$：
    -  $B \times H \times S \times S$  ;  $B \times H \times S \times d$    $\rightarrow B \times H \times S \times d$，每个token根据其对序列内其它tokens的attention scores，来加权序列tokens的embeddings，得到每个token对应的上下文编码向量。
    -  reshape后的形状为，$B \times H \times D$， $D=S \times d$。

- **BertSelfOutput**

  - $\text{O}^{'}_1=\text{LayerNorm}(X + W_0 \cdot \text{Multi-Head}(X))$, **self-connection**, $ B \times S  \times D$

##### BertIntermediate

- $F=\text{GELU}(W_1 \cdot O_1^{'})$， $B \times S \times I$, 其中$W_1 \in \mathbb{R}^{D \times I}$, $I$默认值为3072，用到了gelu激活函数。

##### BertOutput

- $O^{''}_1 =\text{LayerNorm}(O_1^{'}+ W_{2} \cdot F)$, $B \times S \times D$ ，其中，$W_2 \in \mathbb{R}^{I \times D}$.

上述输出$O^{''}_1$作为下一个BertLayer的输入，输出$O^{''}_2$，依次类推，进行迭代，最终输出$O=O^{''}_{12}$，即共12层BertLayer。

### BertPooler

第三步，step3， 获取sequence-level embedding。

拿到上述BertEncoder的输出$O$，shape为$B \times S \times D$，其中每个样本序列(S维度)的第一个token为[CLS]标识的hidden state，标识为$o$，即：$B \times D$。则得到序列级别的嵌入表征：$\text{pooled-sentence-enocding}=\text{tanh}(W \cdot o)$，shape为$B \times D$。这个主要用于下游任务的fine-tuning。

```python
def forward(self, hidden_states):
    # We "pool" the model by simply taking the hidden state corresponding
    # to the first token.
    first_token_tensor = hidden_states[:, 0]
    pooled_output = self.dense(first_token_tensor)
    pooled_output = self.activation(pooled_output) ## nn.tanh
    return pooled_output
```

## Bert Pre-training Tasks

上文介绍了BERT核心的Transformer编码器，下面将介绍Bert的预训练任务。

### BertForMaskedLM

Bert Model with **a language modeling head** on top。上述介绍了BertModel的源码，BertModel主要用于获取序列的编码。本部分要介绍的BertForMaskedLM将基于BertModel得到的序列编码，利用MaskedLM预训练任务进行预训练。

Bert主要利用了Transformer的Encoder，基于encoder得到的序列编码进行预训练，而MLM使得encoder能够进行双向的self-attention。

**BertForMaskedLM**的构造函数：

```python
def __init__(self, config):
    super().__init__(config)
    assert (
    not config.is_decoder
    ), "If you want to use `BertForMaskedLM` make sure `config.is_decoder=False` for bi-directional self-attention." # is_decoder为False，不需要用到decoder

    self.bert = BertModel(config) # BertModel进行序列编码
    self.cls = BertOnlyMLMHead(config) # 多分类预训练任务, task-specific head
    self.init_weights()
```

核心Forward代码：

```python
def forward(self, input_ids=None, attention_mask=None, 
            token_type_ids=None,position_ids=None, 
            head_mask=None, inputs_embeds=None, labels=None,
            encoder_hidden_states=None, encoder_attention_mask=None,
            output_attentions=None, output_hidden_states=None,
            **kwargs):
   
    # step 1: obtain sequence encoding by BertModel
    outputs = self.bert(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
    )

    sequence_output = outputs[0] # B x S x D
    
    # step 2: output scores of each token in the sequence
    prediction_scores = self.cls(sequence_output) # B x S x V, 输出词典中每个词的预测概率

    outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

    # step 3: build loss, label, B x S
    if labels is not None:
        loss_fct = CrossEntropyLoss()  # -100 index = padding token
        masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)) # 拍扁， (B*S) x V
        outputs = (masked_lm_loss,) + outputs

    return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)
```

参数基本上和BertModel一模一样，多了一个labels参数，主要用于获取MLM loss。

其中，cls对应的**BertOnlyMLMHead**类 (其实就是类**BertLMPredictionHead**) 做的主要事情如下公式，即：MLM多分类预测任务，其中$E$为BertModel得到的sequence-token-level encoding，shape为$B \times S \times D$。
$$
\text{Score} = W_1 \cdot \text{LayerNorm}(\text{GELU}(W_0 \cdot E))
$$
其中，$W_0 \in \mathbb{R}^{D \times D}, W_1^{D \times V}$，$V$为vocab的大小。$\text{Score}$的shape为：$B \times S \times V$。

特别的，label的形式：

**labels** (`torch.LongTensor` of shape `(batch_size, sequence_length)`, optional, defaults to `None`) – Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ..., config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`

即，不打算预测的，**label设置为-100**。一般只设置[MASK]位置对应的label，其它位置设置成-100。这样只计算了[MASK]待预测位置的token对应的loss。-100实际上是`CrossEntropyLos`的`ignore_index`参数的默认值。

### BertForPreTraining

和BertForMaskedLM类似，多了一个next sentence prediction预训练任务。Bert Model with **two heads on top** as done during the pre-training: a **masked language modeling** head and **a next sentence prediction** (classification) head. 

此部分对应的heads的核心代码为：

```python
class BertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score
```

其中，BertLMPredictionHead和BertForMaskedLM中的BertLMPredictionHead一样，通过这个来得到MLM loss。另外，多了一个seq_relationship，即拿pooled encoding接一个线性二分类层，判断是否是next sentence，因此可以构造得到next-sentence loss。二者Loss相加。

### BertForNextSentencePrediction

Bert Model with a next sentence prediction (classification) head on top。只有上述的seq_relationship head来构造next-sentence loss，不作赘述。

## Bert Fine-tuning Tasks

下面将介绍利用预训练好的Bert对下游任务进行Fine-tuning的方式。下文介绍的fine-tuning任务对应的model，已经在BERT基础上加了task-specific parameters，只需要利用该model，输入task-specific data，然后optimization一下，就能够得到fine-tuned model。

### BertForSequenceClassification

句子级别的任务，sentence-level task。Bert Model transformer with a sequence classification/regression head on top  (a linear layer on top of the pooled output) e.g. **for GLUE tasks.** 

```python
class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) # 类别数量

        self.init_weights()
        
    # forward输入参数和前文介绍的预训练任务一样
    def forward(self, input_ids=None, attention_mask=None,
                token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, labels=None,
                output_attentions=None, output_hidden_states=None):
        
        # step 1: transformer encoding
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states)
        # step 2: use the pooled hidden state corresponding to the [CLS] token
        # B x D
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        # B x N
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
		# step 3: build loss,  labels: (B, )
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
```

看上述代码，非常清晰。先经过BertModel得到encoding，由于是sentence-level classification，直接拿第一个[CLS] token对应的hidden state过一个分类层得到类别的预测分数logits。再基于logits和labels来构造损失函数。这个任务主要用于sentence-level的分类任务，当然也能够用于sentence-pair-level的分类任务。

### BertForMultipleChoice

句子对级别的任务，**sentence-pair**-level task。Bert Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a softmax) e.g. for **RocStories/SWAG tasks.** 

给一个提示prompt以及多个选择choice(其中有1个是对的，其它是错的)，判断其中哪个选择是对的。**输入格式会整成[[prompt, choice0], [prompt, choice1]…]的形式**。bertModel得到的pooled基础上接一个全连接层，输出在每个“句对“[prompt, choice i]上的logits，然后过一个softmax，构造交叉熵损失。

### BertForTokenClassification

token级别的下游任务，token-level task。Bert Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g. for **Named-Entity-Recognition (NER) tasks.** 

```python
def forward(self, input_ids=None, attention_mask=None,
            token_type_ids=None, position_ids=None, head_mask=None,
            inputs_embeds=None, labels=None,
            output_attentions=None, output_hidden_states=None):    
   # step 1: Transformer
   outputs = self.bert(input_ids, attention_mask=attention_mask,
                       token_type_ids=token_type_ids, position_ids=position_ids,
                       head_mask=head_mask, inputs_embeds=inputs_embeds,
                       output_attentions=output_attentions,
                       output_hidden_states=output_hidden_states)
	# step 2: get sequence-token encoding, B x S x D
    sequence_output = outputs[0]

    # step 3: fine-tuning parameters
    sequence_output = self.dropout(sequence_output)
    # B x S x N
    logits = self.classifier(sequence_output) # nn.Linear(config.hidden_size, config.num_labels)

    outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
    # step 4: build loss, labels, B x S
    if labels is not None:
        loss_fct = CrossEntropyLoss()
        # Only keep active parts of the loss
        if attention_mask is not None:
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss, labels.view(-1), 
                torch.tensor(loss_fct.ignore_index).type_as(labels))
            
            loss = loss_fct(active_logits, active_labels)
         else:
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
         outputs = (loss,) + outputs

    return outputs  # (loss), scores, (hidden_states), (attentions)
```

上述代码一目了然。不作赘述。主要应用于token-level的分类任务，如NER等。

### BertForQuestionAnswering

句子对级别的任务，**sentence-pair**-level task，具体而言，即抽取式问答任务。Bert Model with a **span classification head on top** for extractive question-answering tasks like SQuAD (a linear layers on top of the hidden-states output to compute span start logits and span end logits).

```python
class BertForQuestionAnswering(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        # num_labels为2, 分别代表start_position/end_position对应的下游参数。
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()
        
     # 多了俩参数，start_positions，end_positions，抽取式问答的span label, shape都是(B, )
     def forward(self, input_ids=None, attention_mask=None,
                 token_type_ids=None, position_ids=None,
                 head_mask=None, inputs_embeds=None,
                 start_positions=None, end_positions=None,
                 output_attentions=None, output_hidden_states=None):
            
        # step 1: Transformer encoding
        outputs = self.bert(
            input_ids, # question, passage 
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,)
		# B x S x D
        sequence_output = outputs[0]
        
        # step 2: split to obtain start and end logits
		# B x S x N (N为labels数量,此处N=2)
        logits = self.qa_outputs(sequence_output)
        # split后， B x S x 1, B x S x 1
        start_logits, end_logits = logits.split(1, dim=-1)
        # B x S, B x S
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        
        # step 3: build loss,  start_positions, end_positions: (B, )
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)
			# S 分类
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)
     
```

上述代码主要就是拿sequence-token-level hidden states接两个全连接层，分别输出start_position预测的logits和end_position预测的logits。



## Bert Practice

本部分进行Bert的实践，包括3个部分：

- 利用预训练好的BERT模型，输出目标语句的Embeddings。
- 利用预训练好的BERT模型，预测目标语句中[MASK]位置的真实词。
- 利用预训练好的BERT模型，进行抽取式问答系统。

目前该库实现的预训练模型如下：

- bert-base-chinese
- bert-base-uncased
- bert-base-cased
- bert-base-german-cased
- bert-base-multilingual-uncased
- bert-base-multilingual-cased
- bert-large-cased
- bert-large-uncased
- bert-large-uncased-whole-word-masking
- bert-large-cased-whole-word-masking

上述预训练好的模型的主要差异在于：

- 预训练时的文本语言语料，中文、英文、德文、多语言等
- 有无大小写区分
- 层数
- 预训练时遮盖的是 wordpieces 得到的sub-word 还是整个word

接下来主要采用'bert-base-cased'。在QA部分还会使用上述预训练模型‘bert-large-uncased-whole-word-masking’在SQUAD上的fine-tuning好的模型进行推断。

首先加载**切割器和模型：**

```python
MODEL_NAME = "bert-base-cased"

# step 1: 先获取tokenizer, BertTokenizer, 
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir='tmp/token') 
# step 2: 获取预训练好的模型, BertModel
model = AutoModel.from_pretrained(MODEL_NAME, cache_dir='tmp/model')
```

预览下tokenizer (**transformers.tokenization_bert.BertTokenizer**)：

```python
# 共28996词，包括特殊符号:('[UNK]', 100),('[PAD]', 0),('[CLS]', 101),('[SEP]', 102), ('[MASK]', 103)...
tokenizer.vocab 
```

看下**model**的网络结构：

```python
BertModel(
  (embeddings): BertEmbeddings(
    (word_embeddings): Embedding(28996, 768, padding_idx=0)
    (position_embeddings): Embedding(512, 768)
    (token_type_embeddings): Embedding(2, 768)
    (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
    (dropout): Dropout(p=0.1, inplace=False)
  )
  (encoder): BertEncoder(
    (layer): ModuleList(
      (0): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (1): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (2): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (3): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (4): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (5): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (6): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (7): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (8): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (9): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (10): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
      (11): BertLayer(
        (attention): BertAttention(
          (self): BertSelfAttention(
            (query): Linear(in_features=768, out_features=768, bias=True)
            (key): Linear(in_features=768, out_features=768, bias=True)
            (value): Linear(in_features=768, out_features=768, bias=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
          (output): BertSelfOutput(
            (dense): Linear(in_features=768, out_features=768, bias=True)
            (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
            (dropout): Dropout(p=0.1, inplace=False)
          )
        )
        (intermediate): BertIntermediate(
          (dense): Linear(in_features=768, out_features=3072, bias=True)
        )
        (output): BertOutput(
          (dense): Linear(in_features=3072, out_features=768, bias=True)
          (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
          (dropout): Dropout(p=0.1, inplace=False)
        )
      )
    )
  )
  (pooler): BertPooler(
    (dense): Linear(in_features=768, out_features=768, bias=True)
    (activation): Tanh()
  )
)
```

模型结构参考BertModel源码介绍部分。

### Embeddings produced by pre-trained BertModel 

```python
text = "This is an input example"

# step 1: tokenize, including add special tokens
tokens_info = tokenizer.encode_plus(text, return_tensors="pt") 
for key, value in tokens_info.items():
    print("{}:\n\t{}".format(key, value))
# observe the enriched token sequences
print(tokenizer.convert_ids_to_tokens(tokens_info['input_ids'].squeeze(0).numpy()))

# step 2: BertModel Transformer Encoding
outputs, pooled = model(**tokens_info)
print("Token wise output: {}, Pooled output: {}".format(outputs.shape, pooled.shape))

'''
step 1: outputs:
-----------------------------------------------------------
input_ids:
	tensor([[ 101, 1188, 1110, 1126, 7758, 1859,  102]])
token_type_ids:
	tensor([[0, 0, 0, 0, 0, 0, 0]])
attention_mask:
	tensor([[1, 1, 1, 1, 1, 1, 1]])

['[CLS]', 'This', 'is', 'an', 'input', 'example', '[SEP]']

step 2: outputs:
------------------------------------------------------------
Token wise output: torch.Size([1, 7, 768]), Pooled output: torch.Size([1, 768])
'''

```

### Predict the missing word in a sentence

```python
from transformers import BertForMaskedLM

text = "Nice to [MASK] you" # target token using [MASK] to mask

# step 1: obtain pretrained Bert Model using MLM Loss
maskedLM_model = BertForMaskedLM.from_pretrained(MODEL_NAME, cache_dir='tmp/model')
maskedLM_model.eval() # close dropout

# step 2: tokenize
token_info = tokenizer.encode_plus(text, return_tensors='pt')
tokens = tokenizer.convert_ids_to_tokens(token_info['input_ids'].squeeze().numpy())
print(tokens) # ['[CLS]', 'Nice', 'to', '[MASK]', 'you', '[SEP]']

# step 3: forward to obtain prediction scores
with torch.no_grad():
    outputs = maskedLM_model(**token_info)
    predictions = outputs[0] # shape, B x S x V, [1, 6, 28996]
    
# step 4: top-k predicted tokens
masked_index = tokens.index('[MASK]') # 3
k = 10
probs, indices = torch.topk(torch.softmax(predictions[0, masked_index], -1), k)

predicted_tokens = tokenizer.convert_ids_to_tokens(indices.tolist())
print(list(zip(predicted_tokens, probs)))

'''
output:

[('meet', tensor(0.9712)),
 ('see', tensor(0.0267)),
 ('meeting', tensor(0.0010)),
 ('have', tensor(0.0003)),
 ('met', tensor(0.0002)),
 ('know', tensor(0.0001)),
 ('join', tensor(7.0005e-05)),
 ('find', tensor(5.8323e-05)),
 ('Meet', tensor(2.7171e-05)),
 ('tell', tensor(2.4689e-05))]
'''

```

可以看出，meet的概率最大，且达到了0.97，非常显著。

### Extractive QA

展示sentence-pair level的下游任务。

```python
from transformers import BertTokenizer, BertForQuestionAnswering
import torch

# step 1: obtain pretrained-model in SQUAD
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='tmp/token_qa')
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad', cache_dir='tmp/model_qa')

# step 2: tokenize, sentence-pair, question, passage
question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
encoding = tokenizer.encode_plus(question, text, return_tensors='pt')
input_ids, token_type_ids = encoding["input_ids"], encoding["token_type_ids"]
print(input_ids, token_type_ids)
# observe enriched tokens
all_tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze().numpy())
print(all_tokens)

# step 3: obtain start/end position scores, B x S
start_scores, end_scores = model(input_ids, token_type_ids=token_type_ids) # (B, S)
answer = ' '.join(all_tokens[torch.argmax(start_scores) : torch.argmax(end_scores)+1])
print(answer)
assert answer == "a nice puppet"

'''
output:
step 2:
   input_ids: tensor([[  101,  2040,  2001,  3958, 27227,  1029,   102,  3958, 27227,  2001, 1037,  3835, 13997,   102]]) 
   token_type_ids: tensor([[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]])
   all_tokens:
    ['[CLS]', 'who', 'was', 'jim', 'henson', '?', '[SEP]', 'jim', 'henson', 'was', 'a', 'nice', 'puppet', '[SEP]']   
 
step 3:
   answer:
   a nice puppet
'''
```

可以看出，模型能准确预测出答案，**a nice puppet**。

## Summary

之前一直没有机会阅读BERT源码。这篇文章也算是对BERT源码的一次粗浅的阅读笔记。想要进一步学习的话，可以参考文章，[進擊的 BERT：NLP 界的巨人之力與遷移學習](https://leemeng.tw/attack_on_bert_transfer_learning_in_nlp.html)。总之，基于huggingface提供的transfomers进行二次开发和fine-tune还是比较方便的。下一次会尝试结合AllenNLP，在AllenNLP中使用transformers来解决NLP tasks。



## References

[Transformers: State-of-the-art Natural Language Processing](https://arxiv.org/pdf/1910.03771.pdf)

[深入理解NLP Subword算法：BPE、WordPiece、ULM](https://zhuanlan.zhihu.com/p/86965595)

[huggingface transformers doc](https://huggingface.co/transformers/)

[huggingface transformers source code](https://github.com/huggingface/transformers)

[進擊的 BERT：NLP 界的巨人之力與遷移學習](https://leemeng.tw/attack_on_bert_transfer_learning_in_nlp.html)

[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)