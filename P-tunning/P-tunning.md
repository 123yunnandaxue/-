# GPT Understands, Too
## Introduction
预训练语言模型在近几年取得了巨大的成功，语言模型不仅学习上下文表示，还同时学习到了语法、句法等知识。

针对不同的训练目标，语言模型可以分为三类：以GPT为代表的单向语言模型（unidirectional language models）常用于自然语言生成任务，以Bert为代表的双向语言模型（bidirectional language models）常用于自然语言理解任务，以及综合了前两者的混合语言模型（hybird language models），以XLNet为代表。长时间来，人们都认为GPT这种类型的语言模型不适合自然语言理解任务。

GPT-3只在手工prompt的情况下，在少样本和零样本带来了惊艳的表现。这也证明了，超大的单向语言模型，只要有合适的prompt的情况下，也能在自然语言理解任务上表现很好。但是handcrafted prompt会带来很大的消耗（包括人工和实验成本）。并且如果prompt没设计好，也经常会带来效果的大幅度下降。另外，神经网络本质是连续的，离散的prompt也不是最佳选择。

因此，作者提出了P-tuning，致力于自动寻找连续的prompt，基于寻找到的prompt能够将GPT运用在NLU任务上。

## Motivation
* 语言模型可以在预训练阶段学习到上下文的表征信息，也包括其他方面的知识，例如语法、常识或者世界知识等。
现如今预训练模型包括三种类型，分别是以GPT为代表的单向模型，以BERT为代表的双向模型，以及以XLNet为代表的混合模型（单双向复合）；
* GPT3于2020年下半年提出，其能够在少量样本甚至是0样本情况下获得较好的效果。其主要依赖于提出的新的微调范式（prompt-based和in-context learning）。这表明，即便是单向模型，如果使用合适的人工构建的prompt模板也是可以达到很好的自然语言理解目的；
* GPT模型包含巨大规模的参数，使得其很难被迁移，也很难被落地使用
然而，如何选择handcrafted prompt temporary如同大海捞针，而且需要大量的验证集，同时也可能导致陷入局部最优。先前工作（包括chengdanqi的LM-BFF）致力于解决离散提示模板（discrete prompt temporary）的自动生成问题，然而作者认为，神经网络是连续的，离散的prompt会导致局部最优。
* 同时作者发现，prompt模板发生细微的改变，都会对最终的结果产生戏剧性的变化。例如在knowledge probing任务中，可以生成一个文本提示模板，但是究竟哪一个模板合适？作者发现模板中增删一个token就会产生很大的性能差异。（这一部分与chengdanqi的LM-BFF的发现一样）
* 作者还发现，GPT并非只能生成数据，而无法实现自然语言理解，语言模型是包含许多世界知识和先验知识。
 
综合上述，作者提出了新的方法P-tuning

##  Method
### Architecture
符号定义：给定预训练语言模型M, 输入序列x1:n=x1,x2,...xn，通过模型M的映射层会被映射为输出embedding:e(x0),e(x1),...e(xn)。V代表语言模型M的词汇表。模板为T，[Pi]为第i个prompt token。

1. 传统的做法是：给定模板T={[P0:i],x,[Pi+1:m],y}，传统的离散模板会满足[Pi]∈V，并且将T映射为：{e([P0:i]),e(x),e([Pi+1:m]),e(y)}
2. P-tuning的做法是将[Pi]视为伪token，将模板映射为：{h0,...,hi,e(x),hi+1,...hm,e(y)}，其中hi是可学习的embedding tensors。基于此，能够更好的找到连续的prompt，而不是局限于模型M的词汇表V的表达能力。最后，设下游任务损失函数为L，则最终通过优化下列损失函数去找连续prompt hi: h0:m=arghmin(L(M(x,y)))


P-tuning的具体代码细节可以简单描述为：
* 输入一个句子，以及预先设计的一个离散的模板：The Disney film is good! It was [MASK].；
* 先使用BERT的分词工具分词，并获得input ids、position ids、attention masks等；
* 对输入的template中，挑选一个（或多个）token作为pseudo token：The Disney film is good! [pseudo] was [MASK].其初始化可以直接使用原本的token embedding；
* 对所有的pseudo token Pi ，喂入一层LSTM，并获得每个pseudo token输出的隐状态向量 hi；
* 将整个句子喂入BERT embedding layer，对于pseudo token部分的token embedding，则使用hi，进行替换，最后喂入MLM中获得[MASK]位置的预测结果。

## Optimization
实际上训练连续的prompt仍有如下两个问题

* 不连续性：语言模型M的初始词嵌入e经过预训练后是高度离散的，如果h随机初始化，并且用随机梯度下降算法去更新，会导致其仅在较小的邻域内改变，会陷入局部最小。
* 依赖性：直观上我们会觉得hi应该相互之间互相依赖，而不是互相独立的。因此需要一些特别处理，去关联这些hi。

## 结论
在本文中，我们提出了一种P-Tuning方法，该方法使用连续提示与离散提示串联。P-Tuning提高了性能并稳定了预训练语言模型自适应的训练。在few-shot和全监督设置下，P调优对调优和冻结的语言模型都是有效的


