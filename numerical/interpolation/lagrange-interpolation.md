# 拉格朗日插值法（Lagrange Interpolation Polynomial）

## 定义
对某个多项式函数，已知有给定的 $k + 1$ 个取值点：
$$
\left(x_{0}, y_{0}\right), \ldots,\left(x_{k}, y_{k}\right)
$$
其中 $x_{j}$ 对应着自变量的位置，而 $y_{j}$ 对应着函数在这个位置的取值。

假设任意两个不同的 $x_{j}$ 都互不相同，那么应用拉格朗日插值公式所得到的**拉格朗日插值多项式**为：
$$
L(x):=\sum_{j=0}^{k} y_{j} \ell_{j}(x)
$$

其中每个 $\ell _{j}(x)$ 为**拉格朗日基本多项式**（或称插值基函数），其表达式为：

$$
\ell_{j}(x):=\prod_{i=0, i \neq j}^{k} \frac{x-x_{i}}{x_{j}-x_{i}}=\frac{\left(x-x_{0}\right)}{\left(x_{j}-x_{0}\right)} \cdots \frac{\left(x-x_{j-1}\right)}{\left(x_{j}-x_{j-1}\right)} \frac{\left(x-x_{j+1}\right)}{\left(x_{j}-x_{j+1}\right)} \cdots \frac{\left(x-x_{k}\right)}{\left(x_{j}-x_{k}\right)}
$$
拉格朗日基本多项式 $\ell _{j}(x)$ 的特点是在 $x_{j}$ 上取值为1，在其它的点 $x_{i},\,i\neq j$上取值为0。

## 示例
### 题目
假设有某个二次多项式函数 $f$，已知它在三个点上的取值为：
- $f(4)=10$
- $f(5)=5.25$
- $f(6)=1$
  
求 $f(18)$ 的值。

### 解法

首先写出每个拉格朗日基本多项式：
$$
\begin{aligned}
&\ell_{0}(x)=\frac{(x-5)(x-6)}{(4-5)(4-6)}\\
&\ell_{1}(x)=\frac{(x-4)(x-6)}{(5-4)(5-6)}\\
&\ell_{2}(x)=\frac{(x-4)(x-5)}{(6-4)(6-5)}
\end{aligned}
$$
然后应用拉格朗日插值法，就可以得到 $p$ 的表达式（ $p$为函数 $f$ 的插值函数）：
$$
\begin{aligned}
p(x) &=f(4) \ell_{0}(x)+f(5) \ell_{1}(x)+f(6) \ell_{2}(x) \\
&=10 \cdot \frac{(x-5)(x-6)}{(4-5)(4-6)}+5.25 \cdot \frac{(x-4)(x-6)}{(5-4)(5-6)}+1 \cdot \frac{(x-4)(x-5)}{(6-4)(6-5)} \\
&=\frac{1}{4}\left(x^{2}-28 x+136\right)
\end{aligned}
$$
代入数值 $18$ 就可以求出所需之值：$f(18)=p(18)=-11$。
