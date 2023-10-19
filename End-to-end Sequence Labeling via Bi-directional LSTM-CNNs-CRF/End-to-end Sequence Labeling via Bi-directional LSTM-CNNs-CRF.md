# End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF
## 摘要
现有技术的序列标记系统传统上需要大量手工制作的特征和数据预处理形式的任务特定知识。在本文中，我们介绍了一种新的神经网络架构，`该架构通过使用双向LSTM、CNN和CRF的组合，自动受益于单词和字符级别的表示`。**我们的系统真正是端到端的，不需要特征工程或数据预处理**，因此适用于广泛的序列标记任务。我们在两个序列标记任务的两个数据集上评估了我们的系统——用于词性标记的Penn Treebank WSJ语料库和用于命名实体识别的CoNLL 2003语料库。我们在这两个数据集上都获得了最先进的性能——POS标记的准确率为97.55%，NER的F1准确率为91.21%。
## 介绍
语言序列标记，如词性标记和命名实体识别，是深入理解语言的最初阶段之一，其重要性已在自然语言处理界得到充分认可。<br>
大多数传统的高性能序列标记模型是线性统计模型，包括隐马尔可夫模型（HMM）和条件随机场（CRF），它们严重依赖手工制作的特征和任务特定资源。然而，这种特定任务的知识开发成本高昂（Ma和Xia，2014），使得序列标记模型难以适应新任务或新领域。<br>
在过去的几年里，以分布式单词表示为输入的非线性神经网络（也称为单词嵌入）已被广泛应用于NLP问题，并取得了巨大成功。Collobert等人（2011）提出了一种简单但有效的前馈神经网络，该网络通过在固定大小的窗口内使用上下文来独立地对每个单词的标签进行分类。最近，循环神经网络（RNN）及其变体，如长短期记忆（LSTM）和门控递归单元（GRU），在序列数据建模方面取得了巨大成功。已经提出了几种基于RNN的神经网络模型来解决序列标记任务，如语音识别、POS标记和NER，实现了与传统模型的竞争性能。然⽽，即使是利⽤分布式表示作为输⼊的系统，也利⽤这些表示来增强⽽不是取代⼿⼯制作的特征（例如单词拼写和⼤写字⺟）。`当模型仅依赖于神经编码时，它们的性能会迅速下降`。<br>
在本文中，我们提出了一种用于序列标记的神经网络结构。`它是一个真正的端到端模型，除了在未标记语料库上预训练的单词嵌入之外，不需要特定任务的资源、特征工程或数据预处理。`因此，我们的模型可以很容易地应用于不同语言和领域的广泛序列标记任务。`我们首先使用卷积神经网络（CNNs）将单词的字符级信息编码为其字符级表示。然后，我们将字符和单词级别的表示结合起来，并将它们输入到双向LSTM（BLSTM）中，以对每个单词的上下文信息进行建模。在BLSTM之上，我们使用顺序CRF来联合解码整个句子的标签。`我们在两个语言序列标记任务上评估了我们的模型——宾夕法尼亚树库WSJ上的POS标记和CoNLL 2003共享任务中的英语数据上的NER。我们的端到端模型优于以前的先进系统，POS标记的准确率为97.55%，NER的F1准确率为91.21%。**这项工作的贡献是（i）提出了一种新的用于语言序列标记的神经网络架构。（ii）在两个经典NLP任务的基准数据集上对该模型进行实证评估。（iii）通过这种真正的端到端系统实现最先进的性能。** <br>
## 神经网络结构
在本节中，我们将描述我们的神经网络架构的组件（层）。我们从下到上逐一介绍我们神经网络中的神经层。
### 用于字符级表示的CNN
先前的研究表明，`CNN是从单词的字符中提取形态信息（如单词的前缀或后缀）并将其编码为神经表示的有效方法`。下图显示了我们用来提取给定单词的字符级表示的CNN。CNN与Chiu和Nichols（2015）中的类似，只是我们只使用字符嵌入作为CNN的输入，而没有字符类型特征。在向CNN输入字符编码之前，应用丢弃层。<br>
![](https://github.com/123yunnandaxue/paper-notebook/blob/main/End-to-end%20Sequence%20Labeling%20via%20Bi-directional%20LSTM-CNNs-CRF/picture/14ae2ad5-f839-4cb2-850a-bec6920c6b0f.png)
### 双向LSTM
#### LSTM神经元
循环神经网络（RNN）是一个强大的连接主义模型家族，通过图中的循环来捕捉时间动态。`尽管在理论上，RNN能够捕获长距离依赖关系，但在实践中，由于梯度消失/爆炸问题，RNN失效了。`
`LSTM是RNN的变体，旨在应对这些梯度消失问题。`基本上，LSTM单元由三个乘法门组成，它们控制信息的比例以忘记并传递到下一个时间节点。下图给出了LSTM单元的基本结构。
![](https://github.com/123yunnandaxue/paper-notebook/blob/main/End-to-end%20Sequence%20Labeling%20via%20Bi-directional%20LSTM-CNNs-CRF/picture/7d0948c0-0114-4ab2-a7c6-f6a3ff047bf0.png)<br>
#### BLSTM
对于许多序列标记任务，访问过去（左）和未来（右）的上下文是有益的。然而，LSTM的隐藏状态ht只从过去获取信息，对未来一无所知。一种高效的解决方案是双向LSTM（BLSTM），其有效性已被先前的工作证明。**其基本思想是将每个序列向前和向后呈现为两个独立的隐藏状态，以分别捕获过去和未来的信息。然后将这两个隐藏状态连接起来，形成最终输出。**
### CRF
`对于序列标记（或一般结构化预测）任务，考虑邻域中标签之间的相关性并联合解码给定输入句子的最佳标签链是有益的`。因此，我们使用条件随机场（CRF）联合建模标签序列，而不是独立解码每个标签。
我们使用z={z1,...zn}表示输入的句子，其中zi表示句子中的第i个单词的向量，y={y1,...yn}表示句子的标签。序列CRF的概率模型定义了所有可能的标记序列y上的条件概率p（y|z；W，b），形式如下：
![](https://github.com/123yunnandaxue/paper-notebook/blob/main/End-to-end%20Sequence%20Labeling%20via%20Bi-directional%20LSTM-CNNs-CRF/picture/7b520d44-84f3-4b07-b805-d950558d0c47.png)<br>
然后使用最大似然估计的方法进行参数估计。
### BLSTM-CNNs-CRF
最后，我们通过将BLSTM的输出向量送到CRF层来构建我们的神经网络模型。下图详细说明了我们网络的体系结构。
对于每个单词，第一张图中的CNN以字符编码作为输入来计算字符级别的表示。`然后将字符级表示向量与单词嵌入向量连接，送到BLSTM网络中。最后，BLSTM的输出向量被送到CRF层以联合解码最佳标签序列。丢弃层应用于BLSTM的输入和输出向量`。**实验结果表明，使用dropout显著提高了我们模型的性能。**
![](https://github.com/123yunnandaxue/paper-notebook/blob/main/End-to-end%20Sequence%20Labeling%20via%20Bi-directional%20LSTM-CNNs-CRF/picture/bdc877a9-dc3a-4cb0-95de-a6fc6724e53e.png)
## 网络训练
在本节中，我们将提供有关训练神经网络的详细信息。我们使用Theano库实现了神经网络。单个模型的计算在GeForce GTX TITAN X GPU上运行。
### 参数初始化
 **1.单词编码。**
 我们使用斯坦福大学公开的GloVe 100维编码、Senna 50维编码和Google的Word2Vec 300维编码。为了测验预训练单词编码的效果，我们用有着100为的随机初始化编码进行测试，其中编码是在范围![](https://github.com/123yunnandaxue/paper-notebook/blob/main/End-to-end%20Sequence%20Labeling%20via%20Bi-directional%20LSTM-CNNs-CRF/picture/bcd1d110-4f3d-4cc8-98fc-2903a0c6b0ac.png)内均匀采样的，其中dim是编码的维数。<br>
 **2.字符编码**
 参数矩阵被随机初始化在![](https://github.com/123yunnandaxue/paper-notebook/blob/main/End-to-end%20Sequence%20Labeling%20via%20Bi-directional%20LSTM-CNNs-CRF/picture/bed949a6-0cba-4a59-a342-06020b998234.png)范围内均匀采样。残差向量Bias被初始化为0，但是LSTM中遗忘门的残差向量被初始化为1。<br>
### 最优化算法
参数最优化用的是小批量梯度下降（SGD）算法，批量大小设置为10，动量为0.9。初始化学习率为0.01，学习率的更新公式为：
为了减小梯度爆炸的影响，使用了梯度裁剪，还尝试了其他的更好的最优化算法，如：AdaDelta算法和Adam算法，但是在我们的初步实验中，它们都没有通过动量和梯度裁剪对SGD进⾏有意义的改善。<br>
**早停**使用基于验证集的早停，根据我们的实验最优参数出现在50纪元左右。<br>
**微调**对于每一个纪元的编码，采用微调进行编码初始化，通过反向传播梯度的神经网络模型的梯度更新进行修改。<br>
**Dropout 训练**为了缓解过拟合，采用了丢弃层。应用丢弃层在字符编码被输入到CNN之前以及BLSTM的输入输出向量中。在使用丢弃层之后，模型的结果有了极大的改善。<br>
### 训练超参数
下表总结了我们在所有实验中选择的超参数。<br>
![](https://github.com/123yunnandaxue/paper-notebook/blob/main/End-to-end%20Sequence%20Labeling%20via%20Bi-directional%20LSTM-CNNs-CRF/picture/fd1ae427-2613-4607-9994-5fe0d4ae7ac2.png)<br>
由于时间限制，所以在POS标记和NER上尽可能使用同样的超参数，最后两个任务的超参数几乎都是相同的，除了初始化学习率。
## 实验
### 数据集
我们使用了POS标记和NER两个数据集。<br>
**POS Tagging**该数据集中包含45个不同的POS tags，为了和之前的工作进行比较，将0-18分割为训练集，19-21分给为改良数据，22-24作为测试集。<br>
**NER**使用了BIOES标记而不是标准的BIO2。<br>
### 主要结果
首先，我们运行实验并且通过分割研究去仔细分析每一层的影响性。之后我们和三个基本系统（BRNN\BLSTM\LSTM\BLSTM-CNNs）进行了比较。所有的这些模型都使用了Glove 100维单词编码和相同的超参数。结果如下表所示：<br>
![](https://github.com/123yunnandaxue/paper-notebook/blob/main/End-to-end%20Sequence%20Labeling%20via%20Bi-directional%20LSTM-CNNs-CRF/picture/406ce1e1-440a-4beb-91ce-1b574d8aabb7.png)<br>
通过上表我们可以看出，无论在哪一个任务上BLSTM都比BRNN结果更好，BLSTM-CNN在BLSTM模型的基础上有了进一步的提高，证明了字符级表示对于语言句子表示任务很重要。最后，通过增加CRF层去联合解码，无论在POS还是在NER上都得到了在BLSTM-CNN模型之上的更好的结果。**证明联合解码句子标签对于神经网络模型的最后表现有着极大的好处。**
### 和之前的工作的比较
#### POS标记
下表展示了我们的模型在POS标记上和之前七个具有顶尖表现系统的比较。<br>
![](https://github.com/123yunnandaxue/paper-notebook/blob/main/End-to-end%20Sequence%20Labeling%20via%20Bi-directional%20LSTM-CNNs-CRF/picture/c2195a51-61cf-4b27-b6c5-d9c72abb9a5a.png)<br>
可以看出，我们的模型是最优的。<br>
#### NER
下表展示了我们的模型在NER数据集上F1分数和之前模型的比较。<br>
![](https://github.com/123yunnandaxue/paper-notebook/blob/main/End-to-end%20Sequence%20Labeling%20via%20Bi-directional%20LSTM-CNNs-CRF/picture/2ec436cc-bcae-410d-a591-131b66ddc5bb.png)<br>
通过比较，我们的模型效果最优。<br>
### 单词编码
在之前就提到过，为了测试预训练单词编码的重要性，我们用不同的公开单词编码和一个随机抽样方法进行了实验，结果如下表所示。
![](https://github.com/123yunnandaxue/paper-notebook/blob/main/End-to-end%20Sequence%20Labeling%20via%20Bi-directional%20LSTM-CNNs-CRF/picture/ee1431a8-165a-4925-8161-fd9d491136d4.png)<br>
通过上表可知，比起使用随机编码，模型使用预训练单词编码得到了一个更好的改善。通过比较两个任务，NER会更加依赖预训练编码。
`对于不同的预训练编码，GloVe 100维编码取得了最好的结果。这和Chiu和Nichols (2015)得到的结果不一致,一个可能的原生是Word2Vec的词典不匹配——Word2Vec的编码被训练在情况敏感的行为上，不包括许多常见的符号，例如：标点符号和数字。因为我们没有使用任何数据预处理去处理常见符号或稀有词，这可能会在式样Word2Vec造成问题。`
### Droupout的效果
下表比较了有或没有dropout层的结果。所有超参数和之前的保持一致。证明了droupout层在减少数据过拟合上的效果。<br>
![](https://github.com/123yunnandaxue/paper-notebook/blob/main/End-to-end%20Sequence%20Labeling%20via%20Bi-directional%20LSTM-CNNs-CRF/picture/406ce1e1-440a-4beb-91ce-1b574d8aabb7.png)<br>
### OOV错误分析
为了更好的理解我们的模型的行为，进行了超出词汇表单词（OOV）的错误分析。分割每一个数据集为四个子数据集：IV（在词汇表中的单词）；OOTV（超出训练词汇表的单词）；OOEV（超出编码词汇表的单词）；OOBV（超出词汇表的单词）。如果一个单词出现在训练词汇和编码词汇，则可以认为是IV，如果两者都没有，则是OOBV。OOTV单词没有出现在训练集但是出现在编码词汇表中，OOEV则是没有出现在编码词汇表但是出现在训练集中。下表给出了每一个语料库的统计。<br>
![](https://github.com/123yunnandaxue/paper-notebook/blob/main/End-to-end%20Sequence%20Labeling%20via%20Bi-directional%20LSTM-CNNs-CRF/picture/bcb98641-5f39-4e53-9a07-0af360b15432.png)<br>
上表证明了通过增加CRF层去联合解码，我们的模型在超出训练和编码的数据集中更有效。
## 相关工作
近年来，人们提出了几种不同的神经网络结构，并成功地应用于语言序列标记。在这些神经结构中，与我们的模型最相似的三种方法是Huang等人提出的BLSTM-CRF模型，Chiu和Nichols的LSTM-CNN模型（2015）和Lample等人的BLSTM-GRF（2016）。
1、Huang等人（2015）使用BLSTM进行单词级表示，使用CRF进行联合标记解码，这与我们的模型类似。但他们的模式和我们的模式有两个主要区别。**首先，他们没有使用CNN神经网络来对字符级别的信息进行建模。其次，他们将神经网络模型与手工捕获特征的功能相结合，以提高性能，使他们的模型不是一个端到端的系统。**
2、Chiu和Nichols（2015）提出了一种BLSTM和CNN的混合体来对字符和单词级别的表示进行建模，这与我们模型中的前两层相似。**我们的模型主要与该模型的不同之处在于使用CRF进行联合解码。此外，他们的模型也不是真正的端到端的，因为它利用了外部知识**，如字符类型、大写和词典特征，以及一些专门针对NER的数据预处理（例如，用一个“0”替换所有数字0-9）。
3、Lample等人（2016）提出了一种用于NER的BLSTM-CRF模型，该模型利用BLSTM对字符和单词级别的信息进行建模，并使用与Chiu和Nichols（2015）相同的数据预处理。与之相反的是，**我们使用CNN对字符级信息进行建模，在不使用任何数据预处理的情况下实现了更好的NER性能。**
先前提出了其他几种用于序列标记的神经网络。Labeau等人（2015）提出了一种用于德语POS标记的RNN CNNs模型。该模型类似于Chiu和Nichols（2015）中的LSTM CNNs模型，不同之处在于使用 vanila RNN代替LSTM。另一种采用CNN对字符级信息建模的神经架构是“CharWNN”架构（Santos和Zadrozny，2014），其灵感来自前馈网络（Colobert等人，2011）。CharWNN在英语POS标签上获得了接近最先进的准确性。类似的模型也被应用于西班牙语和葡萄牙语的NER（dos Santos et al.，2015）Ling等人（2015）和Yang等人（2016）也使用BSLTM来组成单词表示的字符嵌入，这与Lample等人类似（2016）。
## 总结
在本文中，我们提出了一种用于序列标记的神经网络结构。`它是一个真正的端到端模型，不依赖于特定任务的资源、特征工程或数据预处理`。与以前最先进的系统相比，我们在两个语言序列标记任务上实现了最先进的性能。未来的工作有几个潜在的方向。首先，我们的模型可以通过探索多任务学习方法来进一步改进，以结合更多有用和相关的信息。例如，我们可以使用POS和NER标签联合训练神经网络模型，以改进在我们的网络中学习的中间表示。另一个有趣的方向是将我们的模型应用于社交媒体（Twitter和微博）等其他领域的数据。由于我们的模型不需要任何领域或任务特定的知识，因此将其应用于这些领域可能很容易。


 

