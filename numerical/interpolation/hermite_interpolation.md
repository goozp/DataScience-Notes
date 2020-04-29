# Hermite Interpolation (埃尔米特插值)

不少实际的插值问题不但要求在节点上的函数值相等，而且还要求对应的导数值也相等,甚至要求高阶导数也相等，满足这种要求的插值多项式就是埃尔米特插值多项式。

常用的埃尔米特插值为二重 Hermite 插值多项式。

## Hermite Polynomials (埃尔米特多项式)

对于 $f \in C^{1}[a, b]$ 且有 $x_{0}, x_{1}, \cdots, x_{n} \in[a, b]$，$f$ 和 $f^{\prime}$ 在 $x_{0}, x_{1}, \cdots, x_{n}$ 的最小次唯一多项式为最高 $2n+1$ 次的埃尔米特多项式：
$$
H_{2 n+1}(x)=\sum_{j=0}^{n} f\left(x_{j}\right) H_{n, j}(x)+\sum_{j=0}^{n} f^{\prime}\left(x_{j}\right) \hat{H}_{n, j}(x)
$$
我们用 $L_{n, j}(x)$ 表示 j 阶的 n 次拉格朗日系数多项式，则其中：
$$
H_{n, j}(x)=\left[1-2\left(x-x_{j}\right) L_{n, j}^{\prime}\left(x_{j}\right)\right] L_{n, j}^{2}(x) \quad \text { and } \quad \hat{H}_{n, j}(x)=\left(x-x_{j}\right) L_{n, j}^{2}(x)
$$

## 误差定理
此外, 如果 $f \in C^{2 n+2}[a, b]$，则
$$
f(x)=H_{2 n+1}(x)+\frac{f^{(2 n+2)}[\xi(x)]}{(2 n+2) !}\left(x-x_{0}\right)^{2} \cdots\left(x-x_{n}\right)^{2}
$$
对于 $(a, b)$ 中的 $\xi(x)$（通常是未知的）。