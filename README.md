# MachineLearning-Notes

机器学习笔记。

包含机器学习、深度学习和一些基础知识的笔记、代码和实例。参考来源已在最后说明。

涉及的库及框架：

- NumPy
- SymPy
- Scikit-learn
- TensorFlow 1.X
- TensorFlow 2.X
- MXNet

## 目录

### 1 - Prerequisite Knowledge (必备知识)

#### 1.1 - Mathematical Foundation and Python Implementation (数学基础和 Python 实现)
- **Vector and Determinant** ([notebook](https://github.com/goozp/MachineLearning-Notes/blob/master/prerequisite/Vector.ipynb)). 向量和行列式
- **Matrix** ([notebook](https://github.com/goozp/MachineLearning-Notes/blob/master/prerequisite/Matrix.ipynb)). 矩阵及其运算
- **Distance** ([notebook](https://github.com/goozp/MachineLearning-Notes/blob/master/prerequisite/Distance.ipynb)). 距离
- **Dirivative** ([notebook](https://github.com/goozp/MachineLearning-Notes/blob/master/prerequisite/Dirivative.ipynb)). 导数，用 SymPy 求导与高阶求导
- **Calculus** ([notebook](https://github.com/goozp/MachineLearning-Notes/blob/master/prerequisite/Calculus.ipynb)). 微积分，用 SymPy 求积分
- **Partial Derivatives** ([notebook](https://github.com/goozp/MachineLearning-Notes/blob/master/prerequisite/Partial-Derivatives.ipynb)). 偏导，用 SymPy 求偏导

### 2 - Basic Models (基础模型)

#### 2.1 - Linear Regression

- **Linear Regression** ([notebook](https://github.com/goozp/mldl-example/blob/master/basic/linear/simple-linear-tf2.ipynb)). TensorFlow 2.X 实现，Keras 自定义模型，简单的线性回归模型
- **Binary Linear Regression** ([notebook](https://github.com/goozp/mldl-example/blob/master/basic/linear/binary-linear-mxnet.ipynb)). MXNet 实现，二元线性回归模型
- **Binary Linear Regression** ([notebook](https://github.com/goozp/mldl-example/blob/master/basic/linear/binary-linear-mxnet-gluon.ipynb)). MXNet Gluon 接口实现，二元线性回归模型

#### 2.2 - Softmax Regression

- **Softmax Regression** ([notebook](https://github.com/goozp/mldl-example/blob/master/basic/softmax/softmax-mxnet.ipynb)). MXNet 实现，Softmax 回归模型，完成 Fashion-MNIST 图像分类任务
- **Softmax Regression** ([notebook](https://github.com/goozp/mldl-example/blob/master/basic/softmax/softmax-mxnet-gluon.ipynb)). MXNet Gluon 接口实现，Softmax 回归模型，完成 Fashion-MNIST 图像分类任务

### 3 - Neural Networks (神经网络)

#### 3.1 - Simple Neural Networks

- **Simple 3-Layer Neural Network** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/simple/3-layer-nn-python.ipynb)). Python/NumPy 实现，一个简单的 3 层神经网络，完成 MNIST 手写体数字图片数据集分类任务
- **Multi-layer Perceptron (MLP)** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/mlp/mlp-tf2.ipynb)). TensorFlow 2.X 实现，多层感知机，MNIST 手写体数字图片数据集分类任务
- **Multi-layer Perceptron (MLP)** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/mlp/mlp-mxnet.ipynb)). MXNet 实现，多层感知机，完成 Fashion-MNIST 图像分类任务

#### 3.2 - Convolution Neural Networks (CNN)

- **Convolution Neural Networks** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/cnn/cnn-tf2.ipynb)). TensorFlow 2.X 实现，卷积神经网络（CNN），MNIST 手写体数字图片数据集分类任务
- **Convolution Neural Networks LeNet** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/cnn/cnn-lenet-mxnet.ipynb)). MXNet 实现，LeNet 卷积神经网络，Fashion-MNIST 图像分类任务
- **Convolution Neural Networks AlexNet** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/cnn/cnn-alexnet-mxnet.ipynb)). MXNet 实现，AlexNet 深度卷积神经网络，Fashion-MNIST 图像分类任务
- **Convolution Neural Networks VGGNet** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/cnn/cnn-vgg-mxnet.ipynb)). MXNet 实现，VGG 深度卷积神经网络，Fashion-MNIST 图像分类任务
- **Convolution Neural Networks - NiN (Network In Network)** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/cnn/cnn-nin-mxnet.ipynb)). MXNet 实现，NiN 卷积神经网络，Fashion-MNIST 图像分类任务
- **Convolution Neural Networks GoogLeNet** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/cnn/cnn-googlenet-mxnet.ipynb)). MXNet 实现，GoogLeNet 卷积神经网络，Fashion-MNIST 图像分类任务
- **Convolution Neural Networks ResNet** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/cnn/cnn-resnet-mxnet.ipynb)). MXNet 实现，ResNet 残差网络，Fashion-MNIST 图像分类任务
- **Convolution Neural Networks DenseNet** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/cnn/cnn-densenet-mxnet.ipynb)). MXNet 实现，DenseNet 稠密连接网络，Fashion-MNIST 图像分类任务

#### 3.3 - Recurrent Neural Networks (RNN)

- **Recurrent Neural Networks** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/rnn/rnn-tf2.ipynb)). TensorFlow 2.X 实现，循环神经网络（RNN），尼采风格文本的自动生成
- **LSTM Recurrent Neural Networks** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/rnn/RNN-LSTM-2-layers-sequential.ipynb)). TensorFlow 2.X，Keras Sequential 实现，LSTM 循环神经网络（RNN），外汇交易（时序数据）预测
- **LSTM Recurrent Neural Networks** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/rnn/RNN-LSTM-2-layers-api.ipynb)). TensorFlow 2.X，Keras 自定义 Model 实现，LSTM 循环神经网络（RNN），外汇交易（时序数据）预测
- **LSTM Bi-directional Recurrent Neural Network**([notebook](https://github.com/goozp/mldl-example/blob/master/nn/rnn/BiRNN-LSTM-2-layers-api.ipynb)). TensorFlow 2.X，Keras 自定义 Model 实现，LSTM 双向循环神经网络（RNN），外汇交易（时序数据）预测
- **GRU Recurrent Neural Networks** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/rnn/RNN-GRU-2-layers-api.ipynb)). TensorFlow 2.X，Keras 自定义 Model 实现，GRU 循环神经网络（RNN），外汇交易（时序数据）预测

### 4 - Application Scenarios (应用场景)

#### 4.1 - Natural Language Processing (NLP, 自然语言处理)

- **Word2Vec (Word Embedding) - Skip-gram** ([notebook](https://github.com/goozp/mldl-example/blob/master/basic/word2vec/skip-gram-tf1.ipynb)). TensorFlow 1.X 实现，Skip-gram 词嵌入模型，维基百科数据

#### 4.2 - Computer Vision (CV, 计算机视觉)

#### 4.3 - Recommended System (RS, 推荐系统)

## Reference

- [《动手学深度学习》](https://zh.d2l.ai/)：所有 MXNet 部分的笔记都整理自这本书
- 《Python 神经网络编程》：引用了手写一个简单神经网络的例子的部分
- 《机器学习中的数学》：基础知识的数学部分
- [简单粗暴 TensorFlow 2.0 | A Concise Handbook of TensorFlow 2.0](https://tf.wiki/)
- [github.com/aymericdamien/TensorFlow-Examples](https://github.com/aymericdamien/TensorFlow-Examples)
