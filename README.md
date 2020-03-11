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

## 目录

### 1 - Prerequisite Knowledge (必备知识)

#### 1.1 - Mathematical Foundation and Python Implementation (数学基础和 Python 实现)

- **Vector and Determinant** ([notebook](https://github.com/goozp/MachineLearning-Notes/blob/master/prerequisite/Vector.ipynb)). 向量和行列式
- **Matrix** ([notebook](https://github.com/goozp/MachineLearning-Notes/blob/master/prerequisite/Matrix.ipynb)). 矩阵及其运算
- **Distance** ([notebook](https://github.com/goozp/MachineLearning-Notes/blob/master/prerequisite/Distance.ipynb)). 距离
- **Dirivative** ([notebook](https://github.com/goozp/MachineLearning-Notes/blob/master/prerequisite/Dirivative.ipynb)). 导数，用 SymPy 求导与高阶求导
- **Calculus** ([notebook](https://github.com/goozp/MachineLearning-Notes/blob/master/prerequisite/Calculus.ipynb)). 微积分，用 SymPy 求积分
- **Partial Derivatives** ([notebook](https://github.com/goozp/MachineLearning-Notes/blob/master/prerequisite/Partial-Derivatives.ipynb)). 偏导，用 SymPy 求偏导

### 2 - Applied Numerical Methods (应用数值方法)

### 3 - Machine Learning Basic Models (机器学习基础模型)

#### 3.1 - Linear Regression

- **Linear Regression** ([notebook](https://github.com/goozp/mldl-example/blob/master/basic/linear/simple-linear-tf2.ipynb)). TensorFlow 2.X 实现，Keras 自定义模型，简单的线性回归模型
- **Binary Linear Regression** ([notebook](https://github.com/goozp/mldl-example/blob/master/basic/linear/binary-linear-mxnet.ipynb)). MXNet 实现，二元线性回归模型
- **Binary Linear Regression** ([notebook](https://github.com/goozp/mldl-example/blob/master/basic/linear/binary-linear-mxnet-gluon.ipynb)). MXNet Gluon 接口实现，二元线性回归模型

#### 3.2 - Softmax Regression

- **Softmax Regression** ([notebook](https://github.com/goozp/mldl-example/blob/master/basic/softmax/softmax-mxnet.ipynb)). MXNet 实现，Softmax 回归模型，完成 Fashion-MNIST 图像分类任务
- **Softmax Regression** ([notebook](https://github.com/goozp/mldl-example/blob/master/basic/softmax/softmax-mxnet-gluon.ipynb)). MXNet Gluon 接口实现，Softmax 回归模型，完成 Fashion-MNIST 图像分类任务

### 4 - Neural Networks and Deep Learning (神经网络和深度学习)

#### 4.1 - Simple Neural Networks (神经网络)

- **Simple 3-Layer Neural Network** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/simple/3-layer-nn-python.ipynb)). Python/NumPy 实现，一个简单的 3 层神经网络，完成 MNIST 手写体数字图片数据集分类任务
- **Multi-layer Perceptron (MLP)** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/mlp/mlp-tf2.ipynb)). TensorFlow 2.X 实现，多层感知机，MNIST 手写体数字图片数据集分类任务
- **Multi-layer Perceptron (MLP)** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/mlp/mlp-mxnet.ipynb)). MXNet 实现，多层感知机，完成 Fashion-MNIST 图像分类任务

#### 4.2 - Convolution Neural Networks (CNN - 卷积神经网络)

- **Convolution Neural Networks** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/cnn/cnn-tf2.ipynb)). TensorFlow 2.X 实现，卷积神经网络（CNN），MNIST 手写体数字图片数据集分类任务
- **Convolution Neural Networks LeNet** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/cnn/cnn-lenet-mxnet.ipynb)). MXNet 实现，LeNet 卷积神经网络，Fashion-MNIST 图像分类任务
- **Convolution Neural Networks AlexNet** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/cnn/cnn-alexnet-mxnet.ipynb)). MXNet 实现，AlexNet 深度卷积神经网络，Fashion-MNIST 图像分类任务
- **Convolution Neural Networks VGGNet** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/cnn/cnn-vgg-mxnet.ipynb)). MXNet 实现，VGG 深度卷积神经网络，Fashion-MNIST 图像分类任务
- **Convolution Neural Networks - NiN (Network In Network)** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/cnn/cnn-nin-mxnet.ipynb)). MXNet 实现，NiN 卷积神经网络，Fashion-MNIST 图像分类任务
- **Convolution Neural Networks GoogLeNet** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/cnn/cnn-googlenet-mxnet.ipynb)). MXNet 实现，GoogLeNet 卷积神经网络，Fashion-MNIST 图像分类任务
- **Convolution Neural Networks ResNet** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/cnn/cnn-resnet-mxnet.ipynb)). MXNet 实现，ResNet 残差网络，Fashion-MNIST 图像分类任务
- **Convolution Neural Networks DenseNet** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/cnn/cnn-densenet-mxnet.ipynb)). MXNet 实现，DenseNet 稠密连接网络，Fashion-MNIST 图像分类任务

#### 4.3 - Recurrent Neural Networks (RNN - 循环神经网络)

- **Recurrent Neural Networks** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/rnn/rnn-tf2.ipynb)). TensorFlow 2.X 实现，循环神经网络（RNN），尼采风格文本的自动生成
- **LSTM Recurrent Neural Networks** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/rnn/RNN-LSTM-2-layers-sequential.ipynb)). TensorFlow 2.X，Keras Sequential 实现，LSTM 循环神经网络（RNN），外汇交易（时序数据）预测
- **LSTM Recurrent Neural Networks** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/rnn/RNN-LSTM-2-layers-api.ipynb)). TensorFlow 2.X，Keras 自定义 Model 实现，LSTM 循环神经网络（RNN），外汇交易（时序数据）预测
- **LSTM Bi-directional Recurrent Neural Network**([notebook](https://github.com/goozp/mldl-example/blob/master/nn/rnn/BiRNN-LSTM-2-layers-api.ipynb)). TensorFlow 2.X，Keras 自定义 Model 实现，LSTM 双向循环神经网络（RNN），外汇交易（时序数据）预测
- **GRU Recurrent Neural Networks** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/rnn/RNN-GRU-2-layers-api.ipynb)). TensorFlow 2.X，Keras 自定义 Model 实现，GRU 循环神经网络（RNN），外汇交易（时序数据）预测

#### 4.4 Generative Adversarial Networks (GAN - 生成对抗网络)

### 5 - Application Scenarios - Natural Language Processing (应用场景-自然语言处理)

#### 5.1 - Word Embedding ([md](https://github.com/goozp/mldl-example/blob/master/nlp/inter_word_embedding.md))

- **Word2Vec - Skip-gram** ([notebook](https://github.com/goozp/mldl-example/blob/master/nlp/word2vec/skip-gram-tf1.ipynb)). TensorFlow 1.X 实现，Skip-gram 词嵌入模型，维基百科数据
- **Word2Vec - CBOW**
- **GloVe: Gensim Word Vector Visualization of GloVe** ([notebook](https://github.com/goozp/mldl-example/blob/master/nlp/GloVe/glove-gensim.ipynb)). Gensim 工具包读取操作 GloVe 预训练词向量并可视化
- **Using NN Embedding Layer**  ([notebook](https://github.com/goozp/mldl-example/blob/master/nlp/sentiment-analysis/embedding-tf2-keras.ipynb)). TensorFlow 2.0 + Keras Embedding Layer，使用 Word Embedding 完成 IMDB 电影评论情感预测任务
- **Using NN Embedding Layer and Pretrained Embedding Data**  ([notebook](https://github.com/goozp/mldl-example/blob/master/nlp/sentiment-analysis/embedding-tf2-keras-pretrained-glove.ipynb)). TensorFlow 2.0 + Keras Embedding Layer + pretrained GloVe Embedding，使用 Word Embedding 完成 IMDB 电影评论情感预测任务

### 6 - Application Scenarios - Computer Vision (应用场景-计算机视觉)

### 7 - Application Scenarios - Recommended System (应用场景-推荐系统)

### 8 - Application Scenarios - Knowledge Graph (应用场景-知识图谱)

## Reference

- [《动手学深度学习》](https://zh.d2l.ai/)：所有 MXNet 部分的笔记都整理自这本书
- 《Python 神经网络编程》：引用了手写一个简单神经网络的例子的部分
- 《机器学习中的数学》：基础知识的数学部分
- [简单粗暴 TensorFlow 2.0 | A Concise Handbook of TensorFlow 2.0](https://tf.wiki/)
- [github.com/aymericdamien/TensorFlow-Examples](https://github.com/aymericdamien/TensorFlow-Examples)
- 《Deep Learning With Python》: [fchollet/deep-learning-with-python-notebooks](https://github.com/fchollet/deep-learning-with-python-notebooks)
