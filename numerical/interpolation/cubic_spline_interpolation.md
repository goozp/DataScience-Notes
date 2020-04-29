# Cubic Spline Interpolation (三次样条插值)

三次样条插值（Cubic Spline Interpolation）简称 Spline 插值，是通过一系列形值点的一条光滑曲线，数学上通过求解三弯矩方程组得出曲线函数组的过程。

## Cubic Spline (三次样条)

给定一个定义于 $[a, b]$ 的函数 $f$ 和一个节点集合 $a=x_{0}<x_{1}<\cdots<x_{n}=b$，$f$ 的三次样条插值 $S$ 为满足以下条件的方程：

1. $S(x)$ 是一个三次多项式，在区间 $\left[x_{j}, x_{j+1}\right]$ 的每一个 $j=0,1, \cdots, n-1$ 用 $S_{j}(x)$ 表示。
2. 对于所有的 $j=0,1, \cdots, n-1$，$S_{j}\left(x_{j}\right)=f\left(x_{j}\right)$ 且 $S_{j}\left(x_{j+1}\right)=f\left(x_{j+1}\right)$
3. 对于所有的 $j=0,1, \cdots, n-2$，$S_{j+1}\left(x_{j+1}\right)=S_{j}\left(x_{j+1}\right)$  ( 根据 (b) )
4. 对于所有的 $j=0,1, \cdots, n-2$，$S_{j+1}^{\prime}\left(x_{j+1}\right)=S_{j}^{\prime}\left(x_{j+1}\right)$ 
5. 对于所有的 $j=0,1, \cdots, n-2$，$S_{j+1}^{\prime \prime}\left(x_{j+1}\right)=S_{j}^{\prime \prime}\left(x_{j+1}\right)$ 
6. 满足以下一组边界条件之一:
   1. $S^{\prime \prime}\left(x_{0}\right)=S^{\prime \prime}\left(x_{n}\right)=0$（自然边界）
   2. $S^{\prime}\left(x_{0}\right)=f^{\prime}\left(x_{0}\right)$ and $S^{\prime}\left(x_{n}\right)=f^{\prime}\left(x_{n}\right)$（限制边界）
   3. $S^{\prime}\left(x_{0}\right)=S^{\prime}\left(x_{n}\right), S^{\prime \prime}\left(x_{0}\right)=S^{\prime \prime}\left(x_{n}\right)$（周期性边界）

### 自然边界三次样条的唯一性
如果 $f$ 是定义在 $a=x_{0}<x_{1}<\cdots<x_{n}=b$ 的方程，则 $f$ 在节点 $x_{0}, x_{1}, \cdots, x_{n}$ 上拥有唯一自然三次样条插值 $S$。即，满足自然边界条件的样条插值 $S^{\prime \prime}(a)=0$ and $S^{\prime \prime}(b)=0$

限制边界三次样条插值的唯一性可以用类似的方式证明。

> 当用三次样条曲线逼近函数时，通常首选限制边界条件，因此区间的端点处的函数导数必须知道或者被逼近。

