# Natural Language Processing Overview

在尝试剖析 NLP 的知识体系的时候，我发现我很难整理出一套很顺滑的体系。

我们有 Word Embedding，但 Word Embedding 并不是必须的，而且可能在 NLP 任务中同时进行。能用于 Word Embedding 的模型可能用于其他 NLP 任务。

我们有语言模型（Language Model），但是语言模型实际上只是一种计算组成句子一系列词语的概率的模型。而且，语言模型主要有统计语言模型（比如 n-gram 语言模型）和神经网络语言模型（比如 LSTM）等。这又涉及到 NLP 包含了传统机器学习和基于深度学习的 NLP。

在中文分词时，我们又得学习基于规则的词典分词。而后又有基于统计机器学习算法的分词。

所以在 NLP 领域，很多东西在我看来很交叉，像我这种强迫症患者很难以舒服的姿势（自己最好理解的方式）捋顺。所以我以 NLP 相对于的细分领域，或者说任务目标来总结各种模型，穿插学过的模型和知识点进行总结。

NLP 任务分类介绍的部分在这里：[Natural Language Processing Task](https://github.com/goozp/mldl-example/blob/master/nlp/intro_chinese_word_segment.md)