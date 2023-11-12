# DEBERTAV3: IMPROVING DEBERTA USING ELECTRA-STYLE PRE-TRAINING WITH GRADIENT-DISENTANGLED EMBEDDING SHARING
## 摘要
本文提出了一种新的预训练语言模型**DeBERTaV3，该模型通过用替换标记检测 （RTD） 替换掩码语言建模 （MLM） 来改进原始的 DeBERTa 模型**，这是一种样本效率更高的预训练任务。我们的分析表明，ELECTRA中的vanilla embeddin共享会损害训练效率和模型性能，因为判别器和生成器的训练损失会向不同的方向拉动令牌嵌入，从而产生“tugof-war”动态。因此，我们**提出了一种新的梯度解缠嵌入共享方法，避免了拉锯战动力学，提高了训练效率和预训练模型的质量**。我们使用与 DeBERTa 相同的设置预训练了DeBERTaV3，以展示其在各种下游自然语言理解 （NLU） 任务中的卓越性能。以8个任务的GLUE基准测试为例，DeBERTaV3 Large模型的平均得分为91.37%，比DeBERTa高1.37%，比ELECTRA高1.91%，在结构相似的模型中创下了新的SOTA（SOTA）。此外，我们预先训练了一个多语言模型 mDeBERTaV3，并观察到与英语模型相比，与强基线相比有更大的改进。例如，mDeBERTaV3 Base 在 XNLI 上实现了 79.8% 的零样本跨语言准确率，比 XLM-R Base 提高了 3.6%，在此基准上创建了新的 SOTA。我们的模型和代码在 https://github.com/microsoft/DeBERTa 上公开提供。<br>
总结：基于deberta，debertav3的改进在两个方向：
1. 改进MLM：通过使用RTD来替换原始的MLM任务
2. 改进Electra：
    * 原因：鉴别器和生成器将所有的token放到不同的方向，一直在那里拔河
    * 方法：梯度解纠缠embedding来避免拔河
    * 好处：提高训练效率+提升训练模型质量
## 介绍
预训练语言模型 （PLM） 的最新进展为许多自然语言处理 （NLP） 任务创造了新的先进结果。在扩展具有数十亿或数万亿个参数的 PLM 的同时是提高 PLM 容量的一种行之有效的方法，更重要的是探索更节能的方法来构建具有更少参数和更少计算成本的 PLM，同时保持高模型容量。朝着这个方向，有一些工作可以显着提高PLM的效率。
* 第一个是 RoBERTa，它通过更大的批量大小和更多的训练数据提高了模型容量。基于RoBERTa，DeBERTa通过结合分离注意力（一种改进的相对位置编码机制）进一步提高了预训练效率。
* 第二种提高效率的新预训练方法是 ELECTRA 提出的替换令牌检测 （RTD）。与 BERT不同，BERT 使用转换器编码器通过掩码语言建模 （MLM） 预测损坏的令牌，RTD 使用生成器来生成不明确的损坏和鉴别器来区分不明确的令牌与原始输入，类似于生成对抗网络 （GAN）。RTD的有效性也得到了包括CoCo-LM在内的多项工作的验证。

本文中，我们探讨了两种提高预训练DeBERTa效率的方法：
1. 在 ELECTRA 式训练之后，我们`将 DeBERTa 中的 MLM 替换为 RTD`，其中模型被训练为鉴别器，以预测损坏输入中的令牌是原始的还是由生成器替换的。我们表明，使用 RTD 训练的 DeBERTa 明显优于使用 MLM 训练的模型。
2. `新的嵌入共享方法`。在ELECTRA中，鉴别器和生成器共享相同的令牌嵌入。然而，我们的分析表明，嵌入共享会损害训练效率和模型性能，因为判别器和生成器的训练损失将令牌嵌入拉向相反的方向。这是因为生成器和鉴别器之间的训练目标非常不同。用于训练生成器的 MLM 试图将语义上相似的标记拉得很近，而判别器的 RTD 试图区分语义相似的标记并尽可能地拉取它们的嵌入以优化二进制分类精度，从而导致它们的训练目标之间发生冲突。换句话说，这会产生“拔河”动态，从而降低训练效率和模型质量。另一方面，我们表明，当我们在下游任务上微调鉴别器时，对生成器和鉴别器使用分离的嵌入会导致显着的性能下降，这表明嵌入共享的优点，例如，生成器的嵌入有利于产生更好的鉴别器，正如 Clark 等人（2020 年）所论证的那样。 **为了平衡这些权衡，我们提出了一种新的梯度解纠缠嵌入共享（GDES）方法，其中生成器与判别器共享其嵌入，但停止从判别器到生成器嵌入的梯度。这样，我们避免了拉锯战效应，并保留了嵌入共享的好处**。我们通过实证证明，GDES提高了预训练的效率和预训练模型的质量。

## 背景
### Transformer
基于 Transformer 的语言模型由 L 堆叠的 Transformer 块组成。每个模块都包含一个多头自注意力层，后跟一个完全连接的位置前馈网络。标准的自注意力机制缺乏一种自然的方式来编码单词位置信息。因此，现有方法为每个输入词嵌入添加了位置偏差，以便每个输入词都由一个向量表示，其值取决于其内容和位置。位置偏差可以使用绝对位置嵌入来实现或相对位置嵌入。几项研究表明，相对位置表示对于自然语言理解和生成任务更有效。
### DEBERTA
DeBERTa 通过两个新颖的组件改进了 BERT：DA（分离注意力）和增强的掩码解码器。与使用单个向量来表示每个输入词的内容和位置的现有方法不同，DA 机制使用两个单独的向量：一个用于内容，另一个用于位置。同时，DA机制的注意力权重是通过解缠矩阵计算的，包括单词的内容和相对位置。与 BERT 一样，DeBERTa 使用掩码语言建模进行预训练。DA机制已经考虑了上下文词的内容和相对位置，但没有考虑这些词的绝对位置，这在许多情况下对于预测至关重要。DeBERTa 使用增强的掩码解码器，通过在 MLM 解码层添加上下文词的绝对位置信息来改进 MLM。
### ELECTRA
#### MLM

基于 Transformer 的大规模 PLM 通常对大量文本进行预训练，以使用称为 MLM 的自我监督目标学习上下文单词表示。具体来说，给定一个序列 X ，我们通过随机屏蔽其 15% 的标记来将其破坏为 ̃X，然后训练由 θ 参数化的语言模型，通过预测以 ̃X 为条件的屏蔽标记 ̃x 来重建 X
#### RTD
DeBERTaV3 用 RTD 替代了 MLM. RTD 是 ELECTRA 提出的一种预训练任务。RTD 任务包含 generator 和 discriminator 两个 transformer encoders。
* generator θ G ：使用 MLM 进行训练，用于生成替换 masked tokens 的 ambiguous tokens，损失函数如下：<br>
![](https://img-blog.csdnimg.cn/c326d68d231c4c3f88803a2c59ab1796.png#pic_center)<br>
* discriminator θ D :使用 RTD (Replaced Token Detection) 进行训练，用于检测输入序列中由 generator 生成的伪造 tokens。它的输入序列 X ~ 由 generator 的输出序列构造得到：<br>
![](https://img-blog.csdnimg.cn/42290fe38db443c18f0c3d90a9ba70d9.png#pic_center)<br>
对于 generator 生成的序列，如果 token i ii 不属于 masked token，则保留 token i ii，如果 token i ii 属于 masked token，则根据 generator 生成的概率分布采样出一个伪造的 token，最终可以得到 discriminator 的生成序列。discriminator 的损失函数为：
![](https://img-blog.csdnimg.cn/021795e2f4a1460ebe8b8ce2c19ef6fd.png#pic_center)<br>
也就是将 generator 伪造的错误 tokens 看作负样本，真实 tokens 看作正样本进行二分类。
总的损失函数为:![](https://img-blog.csdnimg.cn/4e52a1dcef5143b194cb3ca0b196631a.png#pic_center)
## DebertaV3
### 带RTD的Deberta
由于 ELECTRA 中的 RTD 和 DeBERTa 中的解缠注意力机制已被证明在预训练中具有样本效率，因此我们提出了一种新版本的 DeBERTa，称为 DeBERTaV3，将 DeBERTa 中使用的 MLM 目标替换为 RTD 目标，以结合后者的优势。
### Embedding Sharing （ES）
在 RTD 预训练时，ELECTRA 的 generator 和 discriminator 是共享 token embeddings E 的，因此 E 的梯度为：<br>
![](https://img-blog.csdnimg.cn/d9476784c610401d8d8a98aa9988fc7a.png#pic_center)
这相当于是在进行 multitask learning，但 MLM 倾向于使得语义相近的 tokens 对应的 embed 也比较接近，而 RTD 则倾向于使得语义相近的 tokens 对应的 embed 相互远离 (方便进行区分)，这会使得训练的收敛速度很慢。<br>
![](https://img-blog.csdnimg.cn/e7451aab18834547b205c846bc8c4037.png#pic_center)
### No Embedding Sharing（NES）
为了验证上述猜想，作者实现了不共享 token embeddings 的模型版本。在训练的一个迭代中，先用前向+后向传播 LMLM训练 generator，再用前向+后向传播训练 discriminator：<br>
![](https://img-blog.csdnimg.cn/3d20598b6a3e48b080fb714ee7ae6d49.png#pic_center)<br>
在对 generator 和 discriminator 的 token embed 解耦后可以看到，EG 的 token embed 之间比较接近，而 ED的token embed 之间彼此远离，这证明了之前的猜想<br>
![](https://img-blog.csdnimg.cn/35d0b64457d74932bc11cd79a7f90d82.png#pic_center)<br>
实验也进一步证明，不共享 token embeddings 可以有效提高模型收敛速度<br>
![](https://img-blog.csdnimg.cn/8ec59851c279476e8220d64e46d91296.png#pic_center)<br>
然而，不共享 token embeddings 却损害了模型性能，这证明了 ELECTRA 论文中所说的 ES 的好处，除了 parameter-efficiency 以外，generator embed 能使得 discriminator 更好。
### Gradient-Disentangled Embedding Sharing (GDES)
为了结合 ES 和 NES 各自的优点，作者提出了 GDES. GDES 和 ES 一样，共享了 token embeddings，但只使用 L M L M 而不使用 λ L R T D 去更新 E G ，从而使得训练更加高效的同时还能利用 E G去提升 discriminator 的性能。此外，作者还引入了一个初始化为 zero matrix 的 E Δ 去适配 E G：![](https://img-blog.csdnimg.cn/ac4d1ec9418b4cb4bc8191147f08cd7c.png#pic_center)<br>
在训练的一个迭代中，GDES 先用前向+后向传播 (L M L M ) 训练 generator ，并更新共享的 token embed E G，再用前向+后向传播 (λ L R T D ) 训练 discriminator (只更新 E Δ ，不更新 E G )。模型训练完后，discriminator 最终的 token embed 即为 E G + E Δ。<br>
![](https://img-blog.csdnimg.cn/f4f6a4c376f54f84b1f4817b22daadee.png#pic_center)
## 结论
在本文中，我们**提出了一种基于DeBERTa和ELECTRA组合的语言模型的新预训练范式，这是两种分别使用相对位置编码和替换标记检测（RTD）的先进模型**。**我们表明，简单地将这两个模型结合起来会导致预训练的不稳定性和低效率，这是由于RTD框架中生成器和判别器之间的严重干扰问题，这被称为“拔河”动力学**。为了解决这个问题，我们引入了一种新的嵌入共享范式，称为GDES，这是这项工作的主要创新和贡献。**GDES允许判别器在不干扰生成器梯度的情况下利用生成器嵌入层中编码的语义信息，从而提高预训练效率。GDES定义了一种在RTD框架中的生成器和鉴别器之间共享信息的新方法，可以很容易地应用于其他基于RTD的语言模型**。我们进行了广泛的分析和实验，将GDES与其他替代品进行比较，以验证其有效性。
此外，我们表明，与以前最先进的 （SOTA） 模型相比，带有 GDES 的 DeBERTaV3 在涵盖自然语言理解不同方面的各种 NLU 任务上取得了显着改进。例如，DeBERTaV3 在 XNLI 任务的跨语言传输准确率上领先于其他具有类似架构的模型 1.37% 以上，mDeBERTaV3 在跨语言传输准确率上超越 XLM-Rby 3.6%。这些结果突出了所有 DeBERTaV3 模型的有效性，并将 DeBERTaV3 确立为新的 SOTA 预训练语言模型 （PLM），用于不同模型尺度（即 Large、Base、Small 和 XSmall）的自然语言理解。同时，这项工作清楚地表明了进一步提高模型参数效率的巨大潜力，并为未来参数效率更高的预训练语言模型的研究提供了一些方向。
