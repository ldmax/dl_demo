<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
# 1. 问题简述
使用台湾中央气象局丰原站的真实数据建立模型，给出连续9小时的18种指标，预测第10小时的PM2.5数值。
给出的数据包括3个文件：train.csv/test.csv/sampleSubmission.csv。train.csv的数据用于训练模型，test.csv的数据用于验证模型效果，sampleSubmission.csv是最终提交结果的格式。
注：不考虑特征选择或者特征提取，认为第10次连续测量的PM2.5值只与前9次连续测量的PM2.5值有关。
## 1.1 train.csv数据介绍

x向：日期/测站/测量项目/小时
一共24个小时
y向：每个月前20天，每天有一共18个指标
## 1.2 test.csv数据介绍

x向：编号/测项/连续的9个小时
y向：18个指标
# 2. 模型
只考虑最简单的特征，即认为第10个小时的PM2.5数值只跟前9小时的PM2.5有关，并且考虑线性的模型，那么模型如下：
$$h_\theta(x)=\sum_{i=0}^m\theta_ix_i$$
其中
![](http://latex.codecogs.com/gif.latex?x_0=1)
n=9
# 3. 损失函数
![](http://latex.codecogs.com/gif.latex?L(\\theta)=\\frac{1}{2}\\sum_{i=1}^m\\left[h_\\theta(x^{(i)})-y_i\\right]^2)
# 4. 参数更新表达式
注：
![](http://latex.codecogs.com/gif.latex?\\theta_j)表示参数向量![](http://latex.codecogs.com/gif.latex?\\theta)的第j个向量，
![](http://latex.codecogs.com/gif.latex?\\theta)一共有n+1个分量，n是特征的数量（在这个例子中n=9，所以
![](http://latex.codecogs.com/gif.latex?\\theta)是10维向量）。假设训练数据一共有m笔数据。
![](http://latex.codecogs.com/gif.latex?x_j^{(i)})表示第i个向量的第j个分量。
## 4.1 梯度表达式推导
![](http://latex.codecogs.com/gif.latex?\\frac{\\partial}{\\partial\\theta_j}L(\\theta)=\\sum_{i=1}^m\\left[h_\\theta(x^{(i)})-y_i\\right]\\frac{\\partial}{\\partial\\theta_j}h_\\theta(x^{(i)}))
如果令$$loss_i=h_\theta(x^{(i)})-y_i$$，而由$h(\theta)$的表达式有$$\frac {\partial}{\partial \theta_j} h_\theta(x^{(i)})=x_j^{(i)}$$
那么$$\frac {\partial}{\partial \theta_j} L(\theta)=\sum_{i=1}^m loss_i x_j^{(i)}$$
那么$L(\theta)$对各个参数的偏导数的向量$$\nabla L(\theta)=
\left[ 
\begin{array} {cccc}
x_0^{(1)} & x_0^{(2)} & \cdots & x_0^{(m)}  \\
x_1^{(1)} & x_1^{(2)} & \cdots & x_1^{(m)}  \\
\vdots & \vdots & \ddots & \vdots \\
x_n^{(1)} & x_n^{(2)} & \cdots & x_n^{(m)}
\end {array}
\right] 
\left[ 
\begin{array} {c}
loss_1 \\
loss_2 \\
\vdots \\
loss_m
\end{array} 
\right]
$$
即
$$\nabla L_{\theta}(x)=X^\mathrm{T} \cdot \vec{loss}$$
## 4.2 参数更新表达式
由梯度下降参数更新式
$$\vec{\theta} \leftarrow \vec{\theta} - \eta \nabla L_{\theta}(x)$$
代入上式有
$$\vec{\theta} \leftarrow \vec{\theta} - \eta \cdot X^\mathrm{T} \cdot \vec{loss}$$
# 5. 代码
见linear_regression.py