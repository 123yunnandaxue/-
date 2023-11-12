## 摘要
语言模型预训练带来了显著的性能提升，但仔细比较不同方法具有挑战性。训练的计算成本很高，通常是在不同大小的私有数据集上完成的，而且，正如我们将要展示的，超参数选择对最终结果有重大影响。我们提出了一项 BERT 预训练的复制研究（Devlin 等人，2019 年），该研究仔细测量了许多关键超参数和训练数据大小的影响。我们发现 BERT 的训练明显不足，并且可以匹配或超过之后发布的每个模型的性能。我们最好的模型在 GLUE、RACE 和 SQuAD 上取得了最先进的结果。这些结果凸显了以前被忽视的设计选择的重要性，并引发了对最近报告的改进来源的质疑。我们发布我们的模型和代码。
## 介绍
ELMo（Peters 等人，2018 年）、GPT（Radford 等人，2018 年）、BERT（Devlin 等人，2019 年）、XLM（Lample 和 Conneau，2019 年）和 XLNet（Yang 等人，2019 年）等自我训练方法带来了显着的性能提升，但确定这些方法的哪些方面贡献最大可能具有挑战性。 训练的计算成本很高，限制了可以完成的调优量，并且通常使用不同大小的私有训练数据完成，这限制了我们衡量建模进展效果的能力。我们提出了一项 BERT 预训练的复制研究（Devlin 等人，2019 年），其中包括对 hyperparmeter 调整和训练集大小的影响的仔细评估。我们发现 BERT 的训练明显不足，并提出了一种改进的 BERT 模型训练方法，我们称之为 RoBERTa，它可以匹配或超过所有后 BERT 方法的性能。**我们的修改很简单，包括：（1）训练模型的时间更长，批量更大，数据更多;（2）删除下一句预测目标;（3）较长序列的训练;（4）动态改变应用于训练数据的掩码模式**。我们还收集了一个与其他私人使用的数据集大小相当的大型新数据集 （CC-NEWS），以更好地控制训练集大小的影响。在控制训练数据时，我们改进的训练程序改进了 GLUE 和 SQuAD 上已发布的 BERT 结果。当对其他数据进行更长时间的训练时，我们的模型在公共 GLUE 排行榜上的得分为 88.5，与 Yang 等人（2019 年）报告的 88.4 分相匹配。我们的模型在 4/9 的 GLUE 任务上建立了最先进的新技术：MNLI、QNLI、RTE 和 STS-B。我们还在SQuAD和RACE上匹配最先进的结果。总体而言，我们重新确定 BERT 的掩码语言模型训练目标与其他最近提出的训练目标（例如扰动自回归语言建模）具有竞争力（Yang et al.， 2019）。

综上所述，本文的贡献是：
1. 我们提出了一套重要的BERT设计选择和训练策略，并介绍带来更好的下游任务性能的替代方案;
2. 我们使用了一种新的数据集CCNEWS，并确认使用更多数据进行预训练可以进一步提高下游任务的性能;
3. 我们的训练改进表明，在正确的设计选择下，掩码语言模型预训练与所有其他最近发表的方法相比具有竞争力。我们发布了在 PyTorch 中实现的模型、预训练和微调代码（Paszke 等人，2017 年）。
## 实验设置
### 参数设置
参数设置上基本等同于bert，除了Peak learning rate和warmup steps。max-length=512，和bert不同的是，roberta没有像bert一样为了加速训练，90%的更新时使用缩短后的序列（max-length=128），而是用完整长度的序列。
### 数据
Robert的实验中共涉及了5种不同领域的英文语料，共160G：
1） BOOKCORPUS+English WIKIPEDIA (Bert的语料)。16GB
2)  CC-NEWS CommonCrawl News dataset的英文部分。76GB
3） OPENWEBTEXT WebText语料库的开源再造（来自于Reddit）。38GB
4） STORIES 也来自于CommonCrawl，故事风格的文本子集。31GB
### 评价指标
用的还是常见的GLUE， SQuAD，RACE。
## 训练阶段分析
### 静态掩码和动态掩码
**静态掩码**：原始的bert的掩码是在数据预处理阶段随机Mask，因此，在每个epoch中，mask掉的数据都是一样的，为了防止完全相同，bert采用了复制数据的方法，但本质上仍然是静态掩码。比如：将训练数据复制了十份，对这十份分别mask，因此得到了十种不同的mask结果，在40个epoch的训练过程中，相当于每种mask训练了4次（如果是静态掩码，则只有一种mask训练了40次）。

**动态掩码**：Roberta采取的动态掩码并不是在预处理阶段就对数据进行Mask,而是在每次喂入模型的时候进行mask。这就会使得喂入模型的mask数据基本不同。尤其是当训练的epoch很大时，静态掩码会使得同一份掩码结果多次训练，导致模型会机械地记住这个结果。
### 输入格式和NSP任务
Bert中的NSP任务旨在帮助模型在句子层面有更好的理解，其正例为同一个document的连续segment，负例则为不同document，正负样本各占50%。Roberta的实验部分对NSP任务的效果和不同NSP的表现进行实验对比：

（1）segment-pair + NSP：原始Bert的方式,每个segment可能包含多个sentence，但是总长度小于512；

（2）sentence-pair + NSP: 输入为句子对，相较于segment来说比较短，一般远小于512，因此作者用提升Batch size的方式来提高总的token。

（3）FULL-SENTENCES: 为连续采样的句子，长度不超过512。如果跨document的话，则用separator token分割开。无NSP任务

（4）DOC-SENTENCES: 和FULL-SENTENCES类似，只是不能跨document，因此长度会偏短，同样用提高batch size的方法来解决。

实验结果表明，1）对比segment-pair + NSP和sentence-pair + NSP两种方法可以看出，使用segment效果好于使用sentence，因为sentence较短，模型难以学习到长依赖关系；2）无NSP优于有NSP；3）DOC-SENTENCES优于FULL-SENTENCES，但是由于其batch size变化，不易于与其它实验比较，所以后面实验还是使用FULL-SENTENCES；
### 用更大的batch训练
原始的Bert使用256的batch size和1M的steps进行训练，其计算量等于2K的batch size和125K的steps,也等于8K的batch size和31K的steps。基于三者的计算量相等，作者将这三种参数选择进行对比，发现在bach size为2k时表现最好。但是考虑到并行更容易，作者还是选择8k作为后续实验的batch size。
### Text Encoding
Bert和Roberta采用的都是Byte-Pair Encoding。不同的是Roberta使用的是Bytes而不是Unicode作为subword unit。它可以编码任何输入文本，而不会引入任何“unknown”标记。另外，词表会变得更大，参数也更多。
### 更多steps和additional data
这里Roberta的结构延用了Bert-large的结构(L = 24, H = 1024, A = 16, 355M parameters)，在固定8k为batch size时，进行对比实验可以得出以下结论：

1）Roberta的上面1~4小结的配置下，不加额外数据也比Bert-large表现更好。

2）在训练数据加上additional data之后 效果进一步提升。

3）训练更长时间也会带来收益，更多的steps并没有带来过拟合的问题。
## 相关工作
预训练方法的设计具有不同的训练目标，包括语言建模（Dai and Le， 2015;Peters 等人，2018 年;Howard 和 Ruder，2018 年）、机器翻译（McCann 等人，2017 年）和掩码语言建模（Devlin 等人，2019 年;Lample 和 Conneau，2019 年）。 最近的许多论文都为每个最终任务使用了微调模型的基本配方（Howard and Ruder，2018 年;Radford et al.， 2018），并使用掩码语言模型目标的某种变体进行预训练。 然而，较新的方法通过多任务微调（Dong et al.， 2019）、结合实体嵌入（Sun et al.， 2019）、span prediction（Joshi et al.， 2019）和自回归预训练的多种变体（Song et al.， 2019;Chan 等人，2019 年;Yang 等人，2019 年）。 通过在更多数据上训练更大的模型，通常还可以提高性能（Devlin 等人，2019 年;Baevski 等人，2019 年;Yang 等人，2019 年;Radford等人，2019）。我们的目标是复制、简化和更好地调整 BERT 的训练，作为更好地了解所有这些方法的相对性能的参考点。
## 结论
在预训练 BERT 模型时，我们会仔细评估一些设计决策。 我们发现，通过对模型进行更长时间的训练，在更多的数据上使用更大的批次，可以显着提高性能;删除下一句预测目标;对更长的序列进行训练;并动态更改应用于训练数据的屏蔽模式。我们改进的预训练程序，我们称之为 RoBERTa，在 GLUE、RACE 和 SQuAD 上实现了最先进的结果，而无需对 GLUE 进行多任务微调或为 SQuAD 提供额外数据。这些结果说明了这些以前被忽视的设计决策的重要性，并表明BERT的预训练目标与最近提出的替代方案相比仍然具有竞争力。
