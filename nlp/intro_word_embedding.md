# Word Embedding (词嵌入)

## 什么是词嵌入

ont-hot 编码得到的向量是二进制的、稀疏的（绝大部分元素都是0）、维度很高的（维度大小等于词表中的单词个数）；而词嵌入是低维的浮点数向量（即密集向量，与稀疏向量相对）

词嵌入是从数据中学习得到的。

## 使用词嵌入

获取词嵌入有两种方法：

- 在完成任务（比如文档分类或情感预测）的同时学习词嵌入。一开始是随机的词向量，然后对这些词向量进行学习，其学习方式与学习神经网络的权重相同。
- 在不同于待解决问题的机器学习任务上预计算好词嵌入，然后将其加载到模型中。这些词嵌入叫做预训练词嵌入（pretrained word embedding）。

英语电影评论情感分析模型的完美词嵌入空间，可能不同于英语法律文档分类模型的完美词嵌入空间，因为某些语义关系的重要性因任务而异。因此，合理的做法是对每个新任务都学习一个新的嵌入空间。

例如，如果用 Keras 实现的任务，我们要做的就是学习 Embedding 层的权重；而如果是直接使用 pretrained word embedding 就是直接给 Embedding 赋予已经训练好的权重

> 如何定义 Keras Embedding 层
> ```python
> from keras.layers import Embedding
>
> embedding_layer = Embedding(1000, 64)
> ```
> Keras Embedding 层中，第一个参数是标记的个数（最大单词索引+1），第二个参数是嵌入维度