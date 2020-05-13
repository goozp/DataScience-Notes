# BERT (Bidirectional Encoder Representations for Transformers)

## BERT 介绍
在 BERT 之前，语言模型一般是单向的（通过左边上下文或者右边上下文），而语言理解需要双向。

BERT 已经在超大的数据集（Wikipedia 和 BooksCorpus）上进行了预培训，我们仅需要做特定于任务的微调。BERT 是一种深度双向模型，意味着 BERT 在训练过程中从左右两侧上下文学习。BERT 几乎可用于所有 NLP 任务。

BERT 本质上是 Transformer encoders （只是编码器）的捆绑。BERT 是双向的，它的 self-attention 层在两个方向上都应用。

现在已经有很多基于的 BERT 架构的训练方法和语言模型了，比如 TransformerXL，GPT-2，XLNet，ERNIE2.0，RoBERTa 等。

## BERT 的架构

BERT 是一种多层双向 Transformer 编码器。

Transformer 块（transformer blocks）是 BERT 中的 Transformer Encoders 实现块。

比如：
- BERT base
  - 12 layers (transformer blocks)
  - 12 attention heads
  - 110 million parameters
- BERT Large
  - 24 layers (transformer blocks)
  - 16 attention heads
  - 340 million parameters

这个结构在超大数据集中训练，消耗很大，我们很难自己去训练。

BERT 在预训练时，同时完成以下任务：
1. Masked Language Model：遮住 k%（通常 k 是 15；遮得少增加训练成本，遮得太多会缺少上下文）的 input words，然后预测被遮住的词。
2. Next Sentence Prediction.

## BERT 的微调技术

[BERT for Humans: Tutorial+Baseline](https://www.kaggle.com/abhinand05/bert-for-humans-tutorial-baseline)