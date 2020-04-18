# 泰勒多项式和级数（Taylor Polynomials and Series）
泰勒公式的初衷是用多项式来近似表示函数在某点周围的情况。

例如：指数函数 $e^{x}$ 在 $x = 0$ 的附近可以用以下多项式来近似地表示：

$$
\mathrm{e}^{x} \approx 1+x+\frac{x^{2}}{2 !}+\frac{x^{3}}{3 !}+\cdots+\frac{x^{n}}{n !}
$$

## 泰勒定理（Taylor’s Theorem）
假设 $f \in C^{n}[a, b]$，在 $[a, b]$ 存在 $f^{(n+1)}$（即 n+1 阶导），且 $x_{0} \in[a, b]$。对于所有的 $x \in[a, b]$，存在 $\xi(x) \in(x_{0}, x)$ 满足下式：
$$
f(x)=P_{n}(x)+R_{n}(x)
$$

其中， $P_{n}(x)$ 为：
$$
\begin{aligned}
P_{n}(x) &=f\left(x_{0}\right)+f^{\prime}\left(x_{0}\right)\left(x-x_{0}\right)+\frac{f^{\prime \prime}\left(x_{0}\right)}{2 !}\left(x-x_{0}\right)^{2}+\cdots+\frac{f^{(n)}\left(x_{0}\right)}{n !}\left(x-x_{0}\right)^{n} \\
&=\sum_{k=0}^{n} \frac{f^{(k)}\left(x_{0}\right)}{k !}\left(x-x_{0}\right)^{k}
\end{aligned}
$$

$R_{n}(x)$ 为：
$$
R_{n}(x)=\frac{f^{(n+1)}(\xi(x))}{(n+1) !}\left(x-x_{0}\right)^{n+1}
$$

这里 $P_{n}(x)$ 叫做 $x_{0}$ 在 $f$ 的 n 阶**泰勒多项式（泰勒展开式）**，$R_{n}(x)$ 为 $P_{n}(x)$ 相关联的**余项**。

通过取 $P_{n}(x)$ 的极限 $n \rightarrow \infty$ 得到无穷级数，我们称为 $x_{0}$ 在 $f$ 的**泰勒级数**。 在 $x_{0}=0$ 的情况下，泰勒多项式通常被称为**麦克劳林多项式（Maclaurin polynomial）**，泰勒级数则称为**麦克劳林级数（Maclaurin series）**。

泰勒多项式中的余项（截断误差）是指使用截断或有限求和来逼近无限级数之和所涉及的误差。

## 余项的多种形式
上面的 $R_{n}(x)$ 表示实际上是**拉格朗日型余项**，即：
$$
R_{n}(x)=\frac{f^{(n+1)}(\xi(x))}{(n+1) !}(x-x_{0})^{(n+1)}
$$
其中 $\xi(x) \in(x_{0}, x)$。带有拉格朗日型余项的泰勒公式可以视为拉格朗日微分中值定理的推广。

**皮亚诺型余项**的泰勒公式说明了多项式和函数的接近程度：
$$R_{n}(x)= o\left[(x-x_{0})^{n}\right]$$
也就是说，当 $x$ 无限趋近 $x_{0}$ 时，余项 $R_{n}(x)$ 将会是 $(x-x_{0})^{{n}}$ 的高阶无穷小，或者说多项式和函数的误差将远小于 $(x-x_{0})^{{n}}$。

**积分型余项**：
$$
R_{n}(x)=\int_{x_{0}}^{x} \frac{f^{(n+1)}(t)}{n !}(x-t)^{n} d t
$$
