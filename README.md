# DataScience-Notes

数据科学笔记。数据科学的相关笔记、代码和实例，包含数学、统计、机器学习、深度学习等数据科学基础，以及某些应用场景的实现。

参考来源已在最后说明。

大部分代码都是 Python 的，涉及的库及框架：

- NumPy
- SymPy
- SciPy
- Scikit-learn
- Gensim
- TensorFlow 1.X
- TensorFlow 2.X
- MXNet

部分数值分析代码使用 MATLAB。

> 注：   
> (notebook)：Jupyter Notebook 文件链接   
> (MATLAB)：相应的 MATLAB 代码链接    
> (md)：Markdown 文件链接   
> (link)：外部链接



## 目录

### 1 - Prerequisite Knowledge (必备知识)

#### 1.1 - Basic Concepts Related to Mathematics and Python Implementation (数学相关基础概念和 Python 实现)

- **Vector and Determinant** ([notebook](https://github.com/goozp/MachineLearning-Notes/blob/master/prerequisite/Vector.ipynb)). 向量和行列式
- **Matrix** ([notebook](https://github.com/goozp/MachineLearning-Notes/blob/master/prerequisite/Matrix.ipynb)). 矩阵及其运算
- **Distance** ([notebook](https://github.com/goozp/MachineLearning-Notes/blob/master/prerequisite/Distance.ipynb)). 距离
- **Dirivative** ([notebook](https://github.com/goozp/MachineLearning-Notes/blob/master/prerequisite/Dirivative.ipynb)). 导数，用 SymPy 求导与高阶求导
- **Calculus** ([notebook](https://github.com/goozp/MachineLearning-Notes/blob/master/prerequisite/Calculus.ipynb)). 微积分，用 SymPy 求积分
- **Partial Derivatives** ([notebook](https://github.com/goozp/MachineLearning-Notes/blob/master/prerequisite/Partial-Derivatives.ipynb)). 偏导，用 SymPy 求偏导

#### 1.2 - Python and Related Libraries and Frameworks (Python 和相关的类库、框架)

- **[NumPy](https://numpy.org/)**.
    -  [NumPy 基础：数组和向量化计算](https://www.goozp.com/article/118.html)
- **[TensorFlow](https://www.tensorflow.org/)**. Machine learning platform 
- **[PyTorch](https://pytorch.org/)**. Machine learning framework
- **[MXNet](https://mxnet.apache.org/)**. Deep learning framework

#### 1.3 - Basic Knowledge of Mathematics and Statistics (一些数学、统计基础知识)

- **Taylor Polynomials and Series (泰勒多项式和泰勒级数)** ([md](https://github.com/goozp/mldl-example/blob/master/mathematics/taylor-polynomials.md))

### 2 - Applied Numerical Analysis (应用数值分析)

#### 2.1 - Solving Nonlinear Equations (求解非线性方程)

- **The Bisection Method (二分法)** ([notebook](https://github.com/goozp/mldl-example/blob/master/numerical/iteration/bisection-method.ipynb)) ([MATLAB](https://github.com/goozp/mldl-example/blob/master/numerical/iteration/m/bisection.m)). 二分法求单变量方程近似根。
- **Fixed-point Iteration (不动点迭代法)** ([notebook](https://github.com/goozp/mldl-example/blob/master/numerical/iteration/fixed-point-iteration.ipynb)) ([MATLAB](https://github.com/goozp/mldl-example/blob/master/numerical/iteration/m/fixedpoint.m)). 不动点迭代法求单变量方程近似根。
- **Newton's Method (牛顿法)** ([notebook](https://github.com/goozp/mldl-example/blob/master/numerical/iteration/newtons_method.ipynb)) ([MATLAB](https://github.com/goozp/mldl-example/blob/master/numerical/iteration/m/newton.m)). 牛顿法及其拓展（割线法、试错法）求单变量方程近似根。

#### 2.2 - Interpolation (插值)
- **Lagrange Interpolation Polynomial (拉格朗日插值法)**
- **Neville’s Method (内维尔插值)**
- **Hermite Interpolation (埃尔米特插值)**
- **Cubic Spline Interpolation (三次样条插值)**

#### 2.3 - Numerical Differentiation and Integration (数值微积分)

### 3 - Machine Learning Basics （机器学习基础）

#### 3.1 - Feature Engineering (特征工程) ([md](https://github.com/goozp/mldl-example/blob/master/feature/intro_feature_engineering.md))

- **Feature Enhancement (特征增强)** ([notebook](https://github.com/goozp/mldl-example/blob/master/feature/feature_enhancement.ipynb)). 清洗数据（缺失值处理）、标准化和归一化。
- **Feature Construction (特征构建)** ([notebook](https://github.com/goozp/mldl-example/blob/master/feature/feature_construction.ipynb)). 分类变量处理（填充）和编码（独热编码、标签编码、分箱操作）、扩展数值特征
- **Feature Selection (特征选择)** ([notebook](https://github.com/goozp/mldl-example/blob/master/feature/feature_selection.ipynb)). 模型选择、特征选择
- **Feature Transformation (特征转换)** ([notebook](https://github.com/goozp/mldl-example/blob/master/feature/feature_transformation.ipynb)). 主成分分析（PCA）、线性判别分析（LDA）
- **Feature Learning (特征学习)** ([notebook](https://github.com/goozp/mldl-example/blob/master/feature/feature_learning.ipynb)). 受限玻尔兹曼机（RBM）


### 4 - Machine Learning Models - Supervised Learning (机器学习模型 - 监督学习)

#### 4.1 - Linear Regression (线性回归)

- **Simple Linear Regression** ([notebook](https://github.com/goozp/mldl-example/blob/master/basic/linear/simple-linear-tf2.ipynb)). TensorFlow 2.X 实现，Keras 自定义模型，简单的线性回归模型，回归任务
- **Binary Linear Regression** ([notebook](https://github.com/goozp/mldl-example/blob/master/basic/linear/binary-linear-mxnet.ipynb)). MXNet 实现，二元线性回归模型，回归任务
- **Binary Linear Regression** ([notebook](https://github.com/goozp/mldl-example/blob/master/basic/linear/binary-linear-mxnet-gluon.ipynb)). MXNet Gluon 接口实现，二元线性回归模型，回归任务

#### 4.2 - Logistic Regression (Logistic 回归)
- **Logistic Regression** ([notebook](https://github.com/goozp/mldl-example/blob/master/basic/logistic/logistic_skl.ipynb)). Scikit-learn 实现，Logistic 回归线性分类模型，二分类任务

#### 4.3 - Softmax Regression (Softmax 回归)

- **Softmax Regression** ([notebook](https://github.com/goozp/mldl-example/blob/master/basic/softmax/softmax-mxnet.ipynb)). MXNet 实现，Softmax 回归模型，完成 Fashion-MNIST 图像分类任务，多分类任务
- **Softmax Regression** ([notebook](https://github.com/goozp/mldl-example/blob/master/basic/softmax/softmax-mxnet-gluon.ipynb)). MXNet Gluon 接口实现，Softmax 回归模型，完成 Fashion-MNIST 图像分类任务，多分类任务

#### 4.4 - Perceptron (感知机)

- **Perceptron** ([notebook](https://github.com/goozp/mldl-example/blob/master/basic/perceptron/perceptron.ipynb)). Python 实现，感知机线性分类模型，二分类任务
- **Perceptron** ([notebook](https://github.com/goozp/mldl-example/blob/master/basic/perceptron/perceptron_skl.ipynb)). Scikit-learn 实现，感知机线性分类模型，二分类任务

#### 4.5 - Naive Bayes Classification (朴素贝叶斯分类)

- **Naive Bayes Classification** ([notebook](https://github.com/goozp/mldl-example/blob/master/basic/bayes/naive-bayes_skl.ipynb)). Scikit-learn 中的 GaussianNB + MultinomialNB + BernoulliNB 分类任务

#### 4.6 - Support Vector Machine (SVM 支持向量机)

#### 4.7 - Hidden Markov Model (HMM 隐马尔可夫模型)
- **Hidden Markov Model (Introduction and Python Example)** ([notebook](https://github.com/goozp/mldl-example/blob/master/basic/hmm/hmm.ipynb)). 隐马尔可夫模型 Python 示例


### 5 - Machine Learning Models - Unsupervised Learning (机器学习模型 - 无监督学习)

#### 5.1 - K-Means
- **K-Means** ([notebook](https://github.com/goozp/mldl-example/blob/master/basic/k-means/k-means.ipynb)). Python 实现，K-Means 算法，完成鸢尾花分类任务
- **K-Means** ([notebook](https://github.com/goozp/mldl-example/blob/master/basic/k-means/k-means-digits.ipynb)). Scikit-learn，用 K-Means 算法完成手写体识别任务
- **Color Quantization using K-Means** ([notebook](https://github.com/goozp/mldl-example/blob/master/basic/k-means/k-means-color-quantization.ipynb)). 基于 sklearn 演示用 K-Means 进行色彩压缩。

### 6 - Neural Networks and Deep Learning (神经网络和深度学习)

#### 6.1 - Simple Neural Networks (神经网络)

- **Simple 3-Layer Neural Network** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/simple/3-layer-nn-python.ipynb)). Python/NumPy 实现，一个简单的 3 层神经网络，完成 MNIST 手写体数字图片数据集分类任务
- **Multi-layer Perceptron (MLP)** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/mlp/mlp-tf2.ipynb)). TensorFlow 2.X 实现，多层感知机，MNIST 手写体数字图片数据集分类任务
- **Multi-layer Perceptron (MLP)** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/mlp/mlp-mxnet.ipynb)). MXNet 实现，多层感知机，完成 Fashion-MNIST 图像分类任务

#### 6.2 - Convolution Neural Networks (CNN - 卷积神经网络)

- **Convolution Neural Networks** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/cnn/cnn-tf2.ipynb)). TensorFlow 2.X 实现，卷积神经网络（CNN），MNIST 手写体数字图片数据集分类任务
- **Convolution Neural Networks LeNet** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/cnn/cnn-lenet-mxnet.ipynb)). MXNet 实现，LeNet 卷积神经网络，Fashion-MNIST 图像分类任务
- **Convolution Neural Networks AlexNet** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/cnn/cnn-alexnet-mxnet.ipynb)). MXNet 实现，AlexNet 深度卷积神经网络，Fashion-MNIST 图像分类任务
- **Convolution Neural Networks VGGNet** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/cnn/cnn-vgg-mxnet.ipynb)). MXNet 实现，VGG 深度卷积神经网络，Fashion-MNIST 图像分类任务
- **Convolution Neural Networks - NiN (Network In Network)** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/cnn/cnn-nin-mxnet.ipynb)). MXNet 实现，NiN 卷积神经网络，Fashion-MNIST 图像分类任务
- **Convolution Neural Networks GoogLeNet** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/cnn/cnn-googlenet-mxnet.ipynb)). MXNet 实现，GoogLeNet 卷积神经网络，Fashion-MNIST 图像分类任务
- **Convolution Neural Networks ResNet** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/cnn/cnn-resnet-mxnet.ipynb)). MXNet 实现，ResNet 残差网络，Fashion-MNIST 图像分类任务
- **Convolution Neural Networks DenseNet** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/cnn/cnn-densenet-mxnet.ipynb)). MXNet 实现，DenseNet 稠密连接网络，Fashion-MNIST 图像分类任务

#### 6.3 - Recurrent Neural Networks (RNN - 循环神经网络)

- **Recurrent Neural Networks** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/rnn/rnn-tf2.ipynb)). TensorFlow 2.X 实现，循环神经网络（RNN），尼采风格文本的自动生成
- **LSTM Recurrent Neural Networks** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/rnn/RNN-LSTM-2-layers-sequential.ipynb)). TensorFlow 2.x，Keras Sequential 实现，LSTM 循环神经网络（RNN），外汇交易（时序数据）预测
- **LSTM Recurrent Neural Networks** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/rnn/RNN-LSTM-2-layers-api.ipynb)). TensorFlow 2.x，Keras 自定义 Model 实现，LSTM 循环神经网络（RNN），外汇交易（时序数据）预测
- **LSTM Bi-directional Recurrent Neural Network**([notebook](https://github.com/goozp/mldl-example/blob/master/nn/rnn/BiRNN-LSTM-2-layers-api.ipynb)). TensorFlow 2.x，Keras 自定义 Model 实现，LSTM 双向循环神经网络（RNN），外汇交易（时序数据）预测
- **GRU Recurrent Neural Networks** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/rnn/RNN-GRU-2-layers-api.ipynb)). TensorFlow 2.x，Keras 自定义 Model 实现，GRU 循环神经网络（RNN），外汇交易（时序数据）预测

#### 6.4 - Generative Deeping Learning (生成式深度学习)

- **Text Generation with LSTM** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/generative/text-generation-lstm.ipynb)). TensorFlow 2.x + Keras + LSTM + Softmax Temperature Sampling，完成字符级的尼采风格文本生成任务
- **DeepDream** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/generative/deep_dream_tf2.ipynb)). TensorFlow 2.x 实现 DeepDream

### 7 - Application Scenarios - Natural Language Processing (应用场景-自然语言处理)

#### 7.1 - Overview (概览)

- **Overview Introduction (概览介绍)** ([md](https://github.com/goozp/mldl-example/blob/master/nlp/intro_overview.md))
- **Understanding Word Embedding (理解词嵌入)** ([md](https://github.com/goozp/mldl-example/blob/master/nlp/intro_word_embedding.md))
- **Understanding Language Model (理解语言模型)** ([md](https://github.com/goozp/mldl-example/blob/master/nlp/intro_language_modeling.md))
- **Natural Language Processing Task (NLP 任务)** ([md](https://github.com/goozp/mldl-example/blob/master/nlp/intro_nlp_task.md))

#### 7.2 - Word Embedding (词嵌入) 

- **Word2Vec - Skip-gram** ([notebook](https://github.com/goozp/mldl-example/blob/master/nlp/word2vec/skip-gram-tf1.ipynb)). TensorFlow 1.x 实现，Skip-gram 词嵌入模型，维基百科数据
- **Word2Vec - CBOW**
- **GloVe: Gensim Word Vector Visualization of GloVe** ([notebook](https://github.com/goozp/mldl-example/blob/master/nlp/GloVe/glove-gensim.ipynb)). Gensim 工具包读取操作 GloVe 预训练词向量并可视化
- **Using Word Embedding Example**
  - **Using NN Embedding Layer**  ([notebook](https://github.com/goozp/mldl-example/blob/master/nlp/sentiment-analysis/embedding-tf2-keras.ipynb)). TensorFlow 2.x + Keras Embedding Layer，使用 Word Embedding 完成 IMDB 电影评论情感预测任务
  - **Using NN Embedding Layer and Pretrained Embedding Data**  ([notebook](https://github.com/goozp/mldl-example/blob/master/nlp/sentiment-analysis/embedding-tf2-keras-pretrained-glove.ipynb)). TensorFlow 2.x + Keras Embedding Layer + pretrained GloVe Embedding，使用 Word Embedding 完成 IMDB 电影评论情感预测任务,.m/

#### 7.3 - Natural Language Processing Task (NLP 任务)

- **Text Classification (文本分类)**
  - **Bi-LSTM RNN Model** ([notebook](https://github.com/goozp/mldl-example/blob/master/nlp/lm/bi_lstm_rnn_tf2.ipynb)). TensorFlow 2.X 实现，完成 IMDB 电影评论情感预测任务
- **Machine Translation (机器翻译)**
  - **Seq2seq Model - Neural Machine Translation with Attention** ([notebook](https://github.com/goozp/mldl-example/blob/master/nlp/lm/seq2seq_tf2.ipynb)). TensorFlow 2.X 实现，基于 Attention 机制的 Seq2seq 模型
  - **Transformer Model**

### 8 - Application Scenarios - Computer Vision (应用场景-计算机视觉)

### 9 - Application Scenarios - Recommended System (应用场景-推荐系统)

### 10 - Application Scenarios - Knowledge Graph (应用场景-知识图谱)

## Reference

- [《动手学深度学习》](https://zh.d2l.ai/) Aston Zhang and Zachary C. Lipton and Mu Li and Alexander J. Smola：所有 MXNet 部分的笔记都整理自这本书
- 《Python 神经网络编程》Tariq Rashid：引用了手写一个简单神经网络的例子的部分
- 《机器学习中的数学》孙博：基础知识的数学部分
- [简单粗暴 TensorFlow 2.0 | A Concise Handbook of TensorFlow 2.0](https://tf.wiki/)
- [github.com/aymericdamien/TensorFlow-Examples](https://github.com/aymericdamien/TensorFlow-Examples)
- 《Deep Learning With Python》François Chollet: [fchollet/deep-learning-with-python-notebooks](https://github.com/fchollet/deep-learning-with-python-notebooks)
- 《Python Data Science Handbook》Jake VanderPlas: [https://jakevdp.github.io/PythonDataScienceHandbook/](https://jakevdp.github.io/PythonDataScienceHandbook/)
- [(2019)斯坦福CS224n深度学习自然语言处理课程 by Chris Manning](https://www.bilibili.com/video/BV1Eb411H7Pq)

