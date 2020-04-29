# 内维尔插值

拉格朗日内插法的一个实际困难是误差项很难应用，因此，在完成计算之前，通常不知道所需精度需要的多项式的次数。

## Neville’s Method (内维尔方法)

设 $x_{0}, x_{1}, \cdots, x_{k}$ 为方程 $f$ 上的点，且 $x_{j}$ 和 $x_{i}$ 是其中两个不同的数。则：
$$
P(x)=\frac{\left(x-x_{j}\right) P_{0,1, \cdots, j-1, j+1, \cdots, k}(x)-\left(x-x_{i}\right) P_{0,1, \cdots, i-1, i+1, \cdots, k}(x)}{\left(x_{i}-x_{j}\right)}
$$
是插值于 $f$ 上 $k+1$ 个点 $x_{0}, x_{1}, \cdots, x_{k}$ 的 n 阶拉格朗日多项式。

## Neville’s Iterated Interpolation Algorithm (内维尔迭代插值算法)

评估插值多项式 $P$ 对于方程 $f$ 在 $x$ 中的 $n+1$ 个不同数 $x_{0}, x_{1}, \cdots, x_{n}$:


- **INPUT** 
    - numbers $x, x_{0}, x_{1}, \ldots, x_{n}$;
    - values $f\left(x_{0}\right), f\left(x_{1}\right), \ldots, f\left(x_{n}\right)$ as the first column $Q_{0,0}, Q_{1,0}, \ldots, Q_{n, 0} \text { of } Q$
- **OUTPUT** 
  - the table $Q$ with $P(x)=Q_{n, n}$

- **Step 1** 
  - For $i=1,2, \ldots, n$
    - $\text { for } j=1,2, \ldots, i$
      - $\text { set } Q_{i j}=\frac{\left(x-x_{i-j}\right) Q_{i, j-1}-\left(x-x_{i}\right) Q_{i-1, j-1}}{x_{i}-x_{i-j}}$
- **Step 2** 
  - **OUTPUT** $(Q)$;   
  - **STOP**.
