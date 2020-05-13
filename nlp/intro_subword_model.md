# Subword Model 子词模型

## OOV 问题
OOV（out-of-vocabulary）就是单词不在词汇库里的情况。

以前的 Neural Machine Translation 基本上都是基于 word 单词作为基本单位的。那么就会存在 OOV 问题，可以说 Subword model 就是希望解决这个问题。

## 解决 OOV 的思路

Subword 就是利用比 word 更基本的组成来建立模型。

像 word2vec 之类的都是基于 word level 的词向量。

下面是一些解决 OOV 的思路。

### Character-Level Model
将字符作为基本单元，建立Character-level model。

但是由于基本单元换为字符后，相较于单词，其输入的序列更长了，使得数据更稀疏且长程的依赖关系更难学习，训练速度也会降低。

### Subword Model （Byte Pair Encoding 与 SentencePiece）

基本单元介于字符与单词之间的模型称作 Subword Model。

选择 Subword 的算法：
- Byte Pair Encoding （BPE）：思路是把经常出现的byte pair用一个新的byte来代替
- wordpiece model （BPE 的变种）

### Hybrid Model
混合模型。大多数情况下还是采用 word level 模型，而只在遇到 OOV 的情况才采用 character level 模型。

## FastText

FastText 就是利用 subword 将 word2vec 扩充，有效的构建 embedding。其基本思路是将每个 word 表示成 bag of character n-gram 以及单词本身的集合，例如对于 where 这个单词和 n=3 的情况，它可以表示为 `<wh,whe,her,ere,re>`，`<where>` ，其中“<”，“>”为代表单词开始与结束的特殊标记。
