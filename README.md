# MachineLearning-Notes

机器学习笔记。

包含机器学习、深度学习和一些基础知识的笔记、代码和实例。

涉及的库及框架：

- NumPy
- Scikit-learn
- TensorFlow 1.X
- TensorFlow 2.X

## 目录

### 0 - Prerequisite Knowledge (必备基础知识)

- **Vector and Determinant** ([notebook](https://github.com/goozp/MachineLearning-Notes/blob/master/prerequisite/Vector.ipynb)) 向量和行列式
- **Matrix** ([notebook](https://github.com/goozp/MachineLearning-Notes/blob/master/prerequisite/Matrix.ipynb)) 矩阵及其运算

### 1 - Basic Models (基础模型)

#### 1.1 - Linear Regression

- **Linear Regression** ([notebook](https://github.com/goozp/mldl-example/blob/master/basic/linear/simple-linear-tf2.ipynb)) TensorFlow 2.X 实现，Keras 自定义模型，简单的线性回归模型

#### 1.2 - Word2Vec (Word Embedding)

- **Skip-gram** ([notebook](https://github.com/goozp/mldl-example/blob/master/basic/word2vec/skip-gram-tf1.ipynb)) TensorFlow 1.X 实现，Skip-gram 词嵌入模型，维基百科数据

### 2 - Neural Networks (神经网络)

#### 2.1 - Simple Neural Networks

- **Simple 3-Layer Neural Network** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/simple/3-layer-nn-python.ipynb))，Python/NumPy 实现，一个简单的 3 层神经网络，完成 MNIST 手写体数字图片数据集分类任务
- **Multi-layer Perceptron (MLP)** ([notebook](https://github.com/goozp/mldl-example/blob/master/nn/mlp/mlp-tf2.ipynb))，TensorFlow 2.X 实现，多层感知机，MNIST 手写体数字图片数据集分类任务

#### 2.2 - Convolution Neural Networks (CNN)

- **Convolution Neural Networks**([notebook](https://github.com/goozp/mldl-example/blob/master/nn/cnn/cnn-tf2.ipynb))，TensorFlow 2.X 实现，卷积神经网络（CNN），MNIST 手写体数字图片数据集分类任务

#### 2.3 - Recurrent Neural Networks (RNN)

- **Recurrent Neural Networks**([notebook](https://github.com/goozp/mldl-example/blob/master/nn/rnn/rnn-tf2.ipynb))，TensorFlow 2.X 实现，循环神经网络（RNN），尼采风格文本的自动生成
- **LSTM Recurrent Neural Networks**([notebook](https://github.com/goozp/mldl-example/blob/master/nn/rnn/RNN-LSTM-2-layers-sequential.ipynb))，TensorFlow 2.X，Keras Sequential 实现，LSTM 循环神经网络（RNN），外汇交易（时序数据）预测
- **LSTM Recurrent Neural Networks**([notebook](https://github.com/goozp/mldl-example/blob/master/nn/rnn/RNN-LSTM-2-layers-api.ipynb))，TensorFlow 2.X，Keras 自定义 Model 实现，LSTM 循环神经网络（RNN），外汇交易（时序数据）预测
- **LSTM Bi-directional Recurrent Neural Network**([notebook](https://github.com/goozp/mldl-example/blob/master/nn/rnn/BiRNN-LSTM-2-layers-api.ipynb))，TensorFlow 2.X，Keras 自定义 Model 实现，LSTM 双向循环神经网络（RNN），外汇交易（时序数据）预测
- **GRU Recurrent Neural Networks**([notebook](https://github.com/goozp/mldl-example/blob/master/nn/rnn/RNN-GRU-2-layers-api.ipynb))，TensorFlow 2.X，Keras 自定义 Model 实现，GRU 循环神经网络（RNN），外汇交易（时序数据）预测

## Reference

- [github.com/aymericdamien/TensorFlow-Examples](https://github.com/aymericdamien/TensorFlow-Examples)
- 《Python 神经网络编程》
- [简单粗暴 TensorFlow 2.0 | A Concise Handbook of TensorFlow 2.0](https://tf.wiki/)
- 《机器学习中的数学》
