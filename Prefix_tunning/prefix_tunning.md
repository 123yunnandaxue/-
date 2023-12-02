# Prefix-Tuning: Optimizing Continuous Prompts for Generation
## 介绍
Prefix-Tuning即基于提示词前缀优化的微调方法，其原理是在输入token之前构造一段任务相关的virtual tokens（虚拟令牌）作为Prefix，然后训练的时候只更新Prefix部分的参数，而PLM中的其他部分参数固定。

  如下图所示，任务输入是一个线性化的表格（例如，“name: Starbucks | type: coffee shop”），输出是一个文本描述（例如，“Starbucks serves coffee.”）。图中左下红色部分是前缀表示的一系列连续的特定任务向量，也参入注意力计算，类似虚拟的tokens。

  Fine-tuning会更新所有Transformer参数，所以对每个任务都要保存一份微调后的模型权重。而Prefix Tuning仅更新前缀部分的参数，这样不同任务只需要保存不同的前缀，微调成本更小。

![](https://img-blog.csdnimg.cn/169ebadacf50473da13e54c051185f5d.png#pic_center)

## Method
Prefix Tuning优化了前缀的所有层，比需要匹配实际单词嵌入的离散提示更具有表达力。Prefix的优化效果将向上传播到所有Transformer激活层，并向右传播到所有后续的标记，实验也显示Prefix效果优于infix（中缀）。此外，这种方法比干预所有激活层（第7.2节）更简单，避免了长距离依赖，并包含了更多可调参数（表达能力discrete prompting< embedding-only ablation < prefix-tuning）。

针对不同的模型结构，需要构造不同的Prefix：

* 自回归模型：在句子前面添加前缀，得到 z = [PREFIX; x; y]，合适的上文能够在固定 LM 的情况下去引导生成下文（比如GPT3的上下文学习）。
* 编码器-解码器模型：Encoder和Decoder都增加了前缀，得到 z = [PREFIX; x; PREFIX0; y]。Encoder端增加前缀是为了引导输入部分的编码，Decoder 端增加前缀是为了引导后续token的生成。


  为了防止直接更新Prefix的参数导致训练不稳定和性能下降的情况，将Prefix部分通过前馈网络进行映射。在训练过程中，优化Pθ 和FFN的参数。训练结束后，推理时只需要P θ ，而可以舍弃FFN。

## 实验总结

1. 消融实验证实，只在embedding层加入Prefix效果不够好，因此，在每层都加了prompt的参数，改动较大。
2. 在数据稀缺的情况下，前缀的初始化方式对性能有很大的影响。
    * 随机初始化：前缀向量可以随机初始化为固定维度的向量。这种方法适用于一些简单的任务或数据较少的情况。

    * 预训练：前缀向量可以通过预训练的语言模型进行初始化。例如，可以使用预训练的BERT或GPT模型，将其输出的某些层的隐藏状态作为前缀向量。

    * 任务特定训练：前缀向量可以通过在特定任务上进行训练来获得。可以使用任务相关的数据对前缀进行有监督或自监督学习，以获得更具任务相关性的前缀向量。

从下图可见：随机初始化会导致性能较低且变化较大，将前缀初始化为真实单词的激活值可以显著提高生成效果
使用与任务相关的单词（例如“summarization”和“table-to-text”）进行初始化的性能略优于与任务无关的单词（例如“elephant”和“divide”）
 
![](https://img-blog.csdnimg.cn/3fa6ac2466664047a020cf5bddf14ab4.png)

总结：在transformer的每一层都添加Prefix表示的soft prompt，为了保持训练稳定，输入前将其用FFN层进行映射。训练时只更新Prefix部分的参数。
