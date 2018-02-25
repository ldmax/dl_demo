# 1. 问题背景
1994年美国人口普查数据库中提取了一些数据，用于预测一个人的年收入是否>50k，training dataset见train.csv。
数据集的每个观测都是一笔数据，label是income，取值为<=50k或者>50k。feature有14个，分别为：
* age: 年龄，连续型，取值为正整数
* workclass: 工作类别，离散型，取值包括：Private/Self-emp-not-inc/Self-emp-inc/Federal-gov/Local-gov/State-gov/without-pay/Never-worked
* fnlwgt: final weight，即在一个state之内，该条观测代表的人数，取值为正整数
* education: 最高受教育程度，离散型，取值包括：Bachelors/Some-college/11th/HS-grad/Prof-school/Assoc-acdm/Assoc-voc/9th/7th-8th/12th/Masters/1st-4th/10th/Doctorate/5th-6th/Preschool
* education_num: 最高受教育程度代号，取值为正整数
* marital_status: 婚姻状况，离散型，取值包括：Married-civ-spouse/Divorced/Never-married/Separated/Widowed/Married-spouse-absent/Married-AF-spouse（Married-civ-spouse表示配偶是普通公民，Married-AF-spouse表示配偶是Armed Forces）
* occupation: 职业，离散型，取值包括：Tech-support/Craft-repair/Other-service/Sales/Exec-managerial/Prof-specialty/Handlers-cleaners/Machine-op-inspct/Adm-clerical/Farming-fishing/Transport-moving/Priv-house-serv/Protective-serv/Armed-Forces
* relationship: 与他人的（主要）社会关系，离散型，取值包括：Wife/Own-child/Husband/Not-in-family/Other-relative/Unmarried
* race: 种族，离散型，取值包括：White/Asian-Pac-Islander/Amer-Indian-Eskimo/Other/Black
* sex: 性别，离散型，取值包括：Male/Female
* capital_gain: 资本得利，取值为正整数或0
* capital_loss: 资本损失，取值为正整数或0
* hours_per_week: 每周工作小时数，正整数
* native_country: 原生国籍，离散型，取值包括：United-States/Cambodia/England/Puerto-Rico/Canada/Germany/Outlying-US(Guam-USVI-etc)/ India/ Japan/ Greece/ South/ China/ Cuba/ Iran/
Honduras/ Philippines/ Italy/ Poland/ Jamaica/ Vietnam/ Mexico/ Portugal/Ireland/ France/ Dominican-Republic/ Laos/ Ecuador/ Taiwan/ Haiti/ Columbia/Hungary/ Guatemala/Nicaragua/Scotland/ Thailand/ Yugoslavia/ El-Salvador/Trinadad&Tobago/ Peru/ Hong/ Holand-Netherlands

test数据集见test.csv。

问题是使用上述14个feature来预测一条观测所代表的人年收入是否大于50k。
# 2. 模型
用logistic回归来解决这个分类问题。
$$\sigma(z)=\frac {1}{1+e^{-z}}$$
$$P(C_1|x)=\sigma(z)$$
$$z=\sum_{i=0}^n \theta_i x_i$$
其中$x_0=1$
# 3. 似然函数
$$L_\theta(x) = f_\theta(x^{(1)})f_\theta(x^{(2)})\ldots f_\theta(x^{(n)})$$
两边取负的自然对数：
$$-lnL_\theta(x)=-\sum_{i=1}^n\left[\hat y_i lnf_\theta(x^{(i)})+(1-\hat y_i)ln\left(1-f_\theta(x^{(i)})\right)\right]$$
那么要最大化似然函数的值也就变成最小化上式，成为前面在线性回归中熟悉的求最小值问题。沿用梯度下降的思路来求这个最小值。
# 4. 参数更新公式
由梯度下降参数公式
$$\vec{\theta} \leftarrow \vec{\theta} - \eta \nabla L(\theta)$$
这里的
$L(\theta)$是loss function，在本问题中也就是$-lnL_\theta(x)$。
由logistic回归笔记中的推导有：
$$\frac {\partial}{\partial \theta_j} \left[-lnL_\theta(x) \right]=\sum_{i=1}^n-\left(\hat y_i-f_\theta(x^{(i)})\right)x_j^{(i)}$$
其中$f_\theta(x)$就是Sigmoid函数。若记$f_\theta(x^{(i)})-\hat y_i$为$loss_i$，那么上式可以改写为：
$$\frac {\partial}{\partial \theta_j}\left[-lnL_\theta(x) \right]=\sum_{i=1}^n loss_i \cdot x^{(i)}_j$$
$\theta$是(m+1)x1的向量，m是特征个数。x是nxm的矩阵，n是数据条数，m是特征数。那么：
$$\nabla \left[-lnL_\theta(x) \right]=X^\mathrm{T} \cdot \vec{loss}$$
因此参数更新表达式：
$$\vec{\theta} \leftarrow \vec{\theta}-\eta \cdot X^\mathrm{T} \cdot \vec{loss}$$
此处学习速率
$\eta$仍然使用Adagrad算法。
# 5. 代码
见logistic_regression.py
# 6. 讨论
结果并不好，一个可能的原因是没有利用fnlwgt。这个变量的意义是这一条记录在某个州代表的人数。换句话说应该将这个人数作为权重考虑或者将那一条数据重复fnlwgt次，再来train模型。
另外考虑使用scikit-learn来做一下特征方面的工作。






