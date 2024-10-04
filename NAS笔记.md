<h1> NAS论文学习笔记

# 1）NAS with RL - 17 ICLR
**neural architecture search with reinforcement learning**
用强化学习RL架构进行NAS工作

## 贡献点
提出了以RNN作为控制器（Controller）用于NAS的RL架构

## 算法
### 最大化期望奖励
$$
J(\theta_c)=E_{P(a_{1:T};\theta_c)}[R]
$$
概率 $P$ 累乘转为对数累加
$$
\bigtriangledown_{\theta_c}J(\theta_c)=\sum_{t=1}^TE_{P(a_{1:T};\theta_c)}\big[\bigtriangledown_{\theta_c}\log P(a_t|a_{(t-1):1};\theta_c)R\big]
\\
\approx \frac1m\sum_{k=1}^m\sum_{t=1}^T\bigtriangledown_{\theta_c}\log P(a_t|a_{(t-1):1};\theta_c)R_k
$$
为减小方差，引入基线（baseline）$b$:
$$
\frac1m\sum_{k=1}^m\sum_{t=1}^T\bigtriangledown_{\theta_c}\log P(a_t|a_{(t-1):1};\theta_c)\left(R_k - b \right)
$$

### 生成简单CNN层示例
由RNN预测诸如层（filter）、跨步（stride）大小，通道数（channel）等超参数，一层一层从前往后预测
<img src="/assets/image%20copy%2014.png" width="750" height="260">


### 生成跳跃连接
简而言之，每个层（Layer）标记一个锚点（anchor point），每一对层（~~相邻层~~）决定是否形成跳跃连接，其中$W_{prev}$、$W_{curr}$和$v$是可训练的参数，$h_k$是RNN控制器到第$k$层时的隐藏状态（hiddenstate）
$$\mathrm{P}(\mathrm{Layer~j~is~an~input~to~layer~i})=\mathrm{sigmoid}(v^\mathrm{T}\mathrm{tanh}(W_{prev}*h_j+W_{curr}*h_i))$$

<img src="/assets/image%20copy%2015.png" width="750" height="260">


### 并行异步架构，加快训练
分布式架构，多台服务器一起上
<img src="/assets/image%20copy%2013.png" width="750" height="255">
~~就算是并行计算，那也是钱砸出来的，试那么多个模型还是很慢啊~~


### 生成RNN架构
基本原理就是依次生成运算方法或激活函数，最后指定哪个函数归属哪个结点，比如下图中的1,0，那么 $\text{Add,Tanh}$ 就属于 $\text{Tree Index 1}$ ， $\text{ElemMult,ReLU}$ 就属于 $\text{Tree Index 0}$ 。
<img src="/assets/image%20copy%2017.png" width="750" height="280">

划分块示意图如下，可以清楚的看到 $\text{Tree Index} ~i$ 对应的架构，其中 $\text{Cell Inject}$ 指的是上一个细胞状态$C_{t-1}$的处理方式：
<img src="/assets/image%20copy%2018.png" width="350" height="300">

示例为"base 2"架构（实际训练是"base 8"架构）


其中细胞状态$C_t$是借鉴了LSTM的设计原理：
<img src="/assets/image%20copy%2016.png" width="700" height="380">
~~知乎上扒的，有点糊...~~

下图是笔者基于论文RNN设计原理整理的生成LSTM的示意图，至于看上去像是base多少看不太出来，maybe base 4
<img src="/assets/image%20copy%2019.png" width="700" height="510">

## 问题和缺点
大概就是跑的仍然比较慢，不过论文里没挂出来训练多久，只说同时用了800个GPU，都是之后引用的论文的挂出来的（NASNet那篇说500个GPU跑了28天 = 1w4 GPU days）。~~扬长避短~~


------------------------------------------------

# 2）SMASH - 18 ICLR
**SMASH: One-Shot Model Architecture Search through HyperNetworks**
通过构建超网（HyperNet）实现单次模型架构搜索
## 算法：
<img src="/assets/image%20copy%209.png" width="750" height="250">

解释：
1. 首先训练超网$H$（**一个**输入为神经架构，输出为权重的特殊**神经网络**）
- 采样小批量输入$x_i$，随机架构$c$和随机权重$W=H(c)$
- 得到训练误差反向传播更新$H$
2. 随机采样大量架构$c$，找到一个最好的$c_0$
- 在验证集上评估找到最好的$c_0$
3. 正常训练$c_0$的权重，得到结果

## 贡献点
1. 如何采样架构？论文提出了Memory-Bank，将采样拓扑编码为二进制向量
2. 如何采样权重？采用超网（HyperNet）得到二进制拓扑向量所映射的权重空间（其实就是跑一遍超网$H$）


## 如何采样架构
memery-bank读取和写入概念（以ResNet,DenseNet等为例）：
<img src="/assets/image%20copy%2012.png" alt="描述" width="700" height="300">
其中白色矩形为 Memery-Bank ，用于存储神经网络架构（使用预编码规则的二进制向量存储）
（只有一个大Memery-Bank供存储，不是1、3、2个Memery-Bank， ~~笔者起初误解了~~ ）
**读取**：
采样（随机或启发式）网络架构，用于评估、训练或进一步搜索。
**写入**：
生成网络架构（随机、进化算法或其他方法得出），编码后存入。


以CNN为例，论文中采用了下右图的基本网络框架，其中trans为 $\mathbf{1 \times 1卷积 + 平均池化}$ 构建的下采样（downsample）层，卷积的权重由学习得来。
<img src="/assets/image%20copy%2011.png" width="900" height="260">

上左图显示了每个块（block）中的一个操作，它包含一个 $\mathbf{1 \times 1卷积}$ ，然后是可变数量且非线性交错的卷积。

~~实现细节看不太懂~~

## 问题和缺点
不难发现，由于是一次性训练的超网，同一个架构所用的是同一组权重，然而这可能并不合适，对于不同的架构，可能存在不同的特征和需求，使用同一组权重可能无法充分捕捉到每个架构的特性。这可能会导致某些架构的性能不足，尤其是在架构间有显著差异时。

------------------------------------------------

# 3）NASNet - 18 CVPR

**Learning Transferable Architectures for Scalable Image Recognition**
提出了新颖的搜索空间，即常规单元和缩减单元（$\text{Normal Cell and Reduction Cell}$），采用这种搜索空间甚至使得采用强化学习方法只比随机搜索强一点。

## 贡献点
非常好**搜索空间**，常规单元和缩减单元（$\text{Normal Cell and Reduction Cell}$），使得训练得到的网络对数据集分辨率的变化具有较高的鲁棒性，具有可迁移、可扩展性。

## 算法
### 对于缩减单元（$\text{Reduction Cell}$）的解释
常规单元（$\text{Normal Cell}$）是不改变图像特征图的卷积单元，而缩减单元（$\text{Reduction Cell}$）则将特征图的长宽减半，具体操作是将第一个 $\text{operation}$ 的步幅（stride）设置为2。

引用21年kbsAutoML综述（指 $\text{AutoML: A survey of the state-of-the-art}$ ）的图来解释架构：
<img src="/assets/image%20copy%2021.png" width="480" height="600">

$\text{Normal Cell 和 Reduction Cell}$ 的架构相似，每个 $\text{Cell}$ 有 $B$ 个 $\text{Block}$，每个 $\text{Cell}$ 相同 - 缩小搜索空间，每个 $\text{Block}$ 之间不同（当然小概率相同） - 保证灵活性，以下所有搜索方法都是基于 $\text{Block}$ 的。


### 生成 $\mathbf{Block}$ 
NasNet Search Space 大纲图
<img src="/assets/image%20copy%2020.png" width="850" height="450">

$\text{hidden state set}$ ：定义一个隐藏状态集合，在第一个 $\text{Block}$ 的初始状态下，内含两个隐藏状态 $H_1,H_2$ ，对于前两层（即前两个 $\text{Cell}$ ），他们由 $\text{Input Image}$ 产生（~~具体也没说~~）；对于非第一层，他们由前两层产生。

生成 $\text{Block}$ 只需要5个简单步骤：
1. 从 $\text{hidden state set}$ 中挑第一个 $h_1$ （怎么挑？）
2. 从 $\text{hidden state set}$ 中挑第二个 $h_2$
3. 为 $h_1$ 找一个 $\text{operation}_1$（怎么找？）
4. 为 $h_2$ 找一个 $\text{operation}_2$
5. 找一个能够组合 $\text{operation}_1$ 和 $\text{operation}_2$ 的输出来创建一个新 $h_3$ 的方法，随后将 $h_3$ 加入 $\text{hidden state set}$ 

以上“怎么挑”，“怎么找”都是RNN控制器需要训练的内容。

即如下的生成 $\text{one block of a convolutional cell}$ 流程图
<img src="/assets/image%20copy%2022.png" width="960" height="300">

其中候选 $\text{operation}$ 如下：
<img src="/assets/image%20copy%2023.png" width="450" height="150">

**统计**：有 $2$ 种 $\text{Cell}$ ，每个 $\text{Cell}$ 有 $B$ 个 $\text{Block}$ ，每个 $\text{Block}$ 有 $5$ 个 $\text{step}$ ，因此RNN预测器（Controller）总共需做 $\mathbf{2 \times 5B}$ 次预测。

有的同学可能听到这大概懂 $\text{Block}$ 整个设计的流程了，也懂 $\text{Block}$ 的不同是怎么来的了，可要是问**为什么要设计** $\mathbf{hidden~state}$ 呢？

还记得上文提到的灵活性吗，保证灵活性可不只是提高 $\text{Block}$ 的多样性，还有**同一个 $\text{Cell}$ 内部 $\text{Block}$ 的拓扑结构**，即不同 $\text{Block}$ 的连接方式，这可比单单 $\text{Block}$ 的种类多了去了。而一个 $\text{Block}$ 由小巧玲珑的两个输入一个输出组成，因此要找前面的两个层或输入图像（two lower layers or input image），其实就是既结合了不同 $\text{Block}$ 之间的连接多样性，又实现了 $\text{Cell}$ 的残差连接（通过 $\text{Block}_0$ 实现），从而实现了较小的（相比于完全架构搜索（searching for an entire structure），见下图 - 引用自21kbsAutoML综述）搜索空间，妙哉妙哉~
<img src="/assets/image%20copy%2024.png" width="500" height="380">


------------------------------------------------

# 4）ENAS - 18 ICLR
**Efficient Neural Architecture Search via Parameter Sharing**
通过**参数共享**进行高效搜索

这也有一个超网，不过在论文中的表述是 $\text{A Large Computational Graph}$ ，即一张大型计算图$G$，其中任何神经架构都是这张计算图的子图（是Graph，如下图所示，区别于SMASH（Network）），每个边或节点保存的是一个神经架构（卷积或全连接层之类的）的权重，这样保证除了第一遍跑的时候是随机初始化，其余情况一开始直接拿这张大网$G$上的权重跑，这比随机初始化更快收敛，收敛后再在$G$上更新权重。


<img src="/assets/image%20copy%2025.png" width="600" height="320">

## 算法

采用的还是强化学习（RL）方法，其中控制器由一个LSTM组成，它的主要任务是什么呢，如上图所示，对于每个节点，它主要是决定之前需要连哪个节点，还有采样什么激活函数（activation function）。

两组参数**交替**训练：
1. LSTM 控制器（controller）的参数 $\theta$
2. 子模型共享参数（就是那张大型计算图 $G$ 的参数）$\omega$

还是基于 $\text{Cell}$ 的训练

### RNN
Controller对每个结点进行前结点采样和激活函数采样，权重取 $G$ 中的 $\mathbf{W}_{i,j}^\mathbf{h}$（ $\mathbf{h}$ 代表隐藏状态），最后对所有汇点取平均后输出。
<img src="/assets/image%20copy%2026.png" width="960" height="260">

在Penn Treebank数据集上只跑了0.45个 GPU days（一张卡跑了10h）

### CNN
先是试了试基于 $\text{Layer}$ 的，理论可行，但是搜索空间大（候选操作 $M=6$ ，层数 $L=12$ 时，搜索空间大小 $M^L \times 2^{\frac{L(L-1)}{2}}=1.6 \times 10^{29}$ ），收敛太慢

再是基于 $\text{Cell}$ 的，这里引用了NASNet的搜索空间，即引入常规单元和缩减单元（$\text{Normal Cell and Reduction Cell}$）

过程也与之前类似，一个 $\text{Cell}$ 含 $B$ 个 $\text{Block}$， 对于 $\text{node}_i$ （ $2<=i<B$ ），为其挑选两个之前的节点作为输入，再选两个对应的操作（ $\text{operation}$ ），将结果相加后输出。

<img src="/assets/image%20copy%2027.png" width="780" height="400">

缩减单元同理。

候选操作 $M=5$ ，每单元块数 $B=7$ 时，，搜索空间大小 $(M^B\times(B-2)!)^4=7.7 \times 10^{27}$ ，相比 $1.6 \times 10^{29}$ 小多了。（原文是 $M=5,B=7,(M\times(B-2)!)^4 = {1.3\times 10^{11}}$ ，感觉有点问题，应该是 $(M^B\times(B-2)!)^4$ ）

关于 $(M^B\times(B-2)!)^4$ 的解释：
1. 除前两个块外排列组合 $(B-2)!$
2. 每个块两条 $\text{input}$ 的排列组合 $((B-2)!)^2$
3. 每条 $\text{input}$ 都有一个对应的 $\text{operation}$ $(M^B\times(B-2)!)^2$
4. 常规单元和缩减单元独立采样 $(M^B\times(B-2)!)^4$


考虑到不同的 $\text{operation}$ 对应的权重不一样（比如 $3\times3$ 的权重和 $5\times5$ 的权重），无法共用，因此应该使用多个类似于 $\mathbf{W}_{i,j}^\mathbf{sep\_conv\_3\times3}$ ， $\mathbf{W}_{i,j}^\mathbf{conv\_5\times5}$ 的权重，而不是单个 $\mathbf{W}_{i,j}$ 

> **可分离卷积（Separable Convolution）Tips**:
其中 $\mathbf{sep\_conv\_3\times3}$ 代表 $3\times3$ 的可分离卷积，那么啥是分离卷积？
对于图像大小为 $\mathrm{N}$ ，输入、输出通道数分别为 $\mathrm{I}$ 、 $\mathrm{O}$ ，卷积核大小为 $\mathrm{S}$ 
> 1. 传统卷积计算量
$$
\mathrm{N} \times \mathrm{I} \times \mathrm{O} \times \mathrm{S}
$$
> 2. 分离卷积计算量（深度卷积 + 逐点卷积）
> - **深度卷积**： $$\mathrm{N} \times \mathrm{I} \times \mathrm{S}$$
> - **逐点卷积**： $$\mathrm{N} \times \mathrm{I} \times \mathrm{O}$$

In summary, 可分离卷积通过削弱传统卷积的灵活性，大大减少了计算量。


------------------------------------------------

# 5）DARTS - 19 ICLR
**DARTS: differentiable architecture search**
提出了**可微分搜索空间**

## 算法
### DARTS解释
常规强化学习使用的搜索空间是离散的，不连续的，所谓**可微分搜索空间**就是将结点之间的连接通过不同操作加权和得到，通过学习其中的**权**来获取架构，最后由权最大的操作导出最终架构。
<img src="/assets/image%20copy%2028.png" width="800" height="400">

以下是边混合操作的公式：
$$\bar{o}^{(i,j)}(x)=\sum_{o\in\mathcal{O}}\frac{\exp(\alpha_o^{(i,j)})}{\sum_{o'\in\mathcal{O}}\exp(\alpha_{o'}^{(i,j)})}o(x)$$
其中 $\alpha_o^{(i,j)}$ 表示边 $(i,j)$ 上操作 $o$ 的权重

### 双层优化问题
最终目标是要获得一个架构$\alpha$，它能使得在验证集上的损失 $\mathcal{L}_{val}(w^{*}(\alpha),\alpha)$ 最小化，其中权重 $w^{*}(\alpha)$ 是能够使训练集损失 $\mathcal{L}_{train}(w,\alpha)$ 最小化的权重 $w$ ，这是一个双层优化问题（bilevel optimization），形式化表示如下：
$$
\begin{aligned}
&\min_{\alpha}\quad\mathcal{L}_{val}(w^{*}(\alpha),\alpha)
\\
&\mathrm{s.t.}\quad w^{*}(\alpha)=\arg\min_{w} \mathcal{L}_{train}(w,\alpha)
\end{aligned}
$$

### 近似架构梯度
算法如下：
$$\begin{aligned}&\text{Create a mixed operation }\bar{o}^{(i,j)}\text{ parametrized by }\alpha^{(i,j)}\text{ for each edge }(i,j)\\&\textbf{while }not\textit{ converged do}
\\&\begin{array}{c}1.\text{ Update architecture }\alpha\text{ by descending }\nabla_\alpha\mathcal{L}_{val}(w-\xi\nabla_w\mathcal{L}_{train}(w,\alpha),\alpha)
\\(\xi=0\text{ if using first-order approximation})\end{array}
\\&\begin{array}{c}2.\text{ Update weights }w\text{ by descending }\nabla_w\mathcal{L}_{train}(w,\alpha)\end{array}
\\&\text{Derive the final architecture based on the learned }\alpha.\end{aligned}$$

其中 $\nabla_{\alpha}\mathcal{L}_{val}(w-\xi\nabla_{w}\mathcal{L}_{train}(w,\alpha),\alpha)$ 是估算 $\nabla_{\alpha}\mathcal{L}_{val}(w^{*}(\alpha),\alpha)$ 的结果。

估算后经链式法则推导，由于二阶导难以计算，为进一步减少计算量，对二阶导进行近似，推导过程如下：
$$\begin{align}
&\quad\nabla_{\alpha}\mathcal{L}_{val}(w^{*}(\alpha),\alpha) \\
&\approx \nabla_{\alpha}\mathcal{L}_{val}(w-\xi\nabla_{w}\mathcal{L}_{train}(w,\alpha),\alpha) \\
&=\nabla_{\alpha}\mathcal{L}_{val}(w',\alpha)-\xi\nabla_{\alpha,w}^{2}\mathcal{L}_{train}(w,\alpha)\nabla_{w'}\mathcal{L}_{val}(w',\alpha) \\
&\approx\nabla_{\alpha}\mathcal{L}_{val}(w^{\prime},\alpha)- \xi\frac{\nabla_{\alpha}\mathcal{L}_{train}(w^{+},\alpha)-\nabla_{\alpha}\mathcal{L}_{train}(w^{-},\alpha)}{2\epsilon}
\end{align}$$

其中 $w'=w-\xi\nabla_{w}\mathcal{L}_{train}(w,\alpha)$ ， $w^{\pm}=w\pm\epsilon\nabla_{w^{\prime}}\mathcal{L}_{val}(w^{\prime},\alpha)$

$(3)$ 式到 $(4)$ 式用到了简单的泰勒展开：
$$
\begin{aligned}
f(x_0+h)&\approx f(x_0)+\frac{f'(x_0)}{1!}(x_0+h-x_0) \\
&=f(x_0)+f'(x_0)\cdot h
\end{aligned}
$$

变形得：
$$
\begin{aligned}
f'(x_0)\approx \frac{f(x_0+h)-f(x_0)}{h}
\end{aligned}
$$

同理：
$$
\begin{aligned}
f'(x_0)\approx \frac{f(x_0)-f(x_0-h)}{h}
\end{aligned}
$$

上下式相加除二得：
$$
\begin{aligned}
f'(x_0)\approx \frac{f(x_0+h)-f(x_0-h)}{2h}
\end{aligned}
$$

令$h=\epsilon\cdot A$，得到：
$$
\begin{aligned}
f'(x_0)\cdot A\approx \frac{f(x_0+\epsilon\cdot A)-f(x_0-\epsilon\cdot A)}{2\epsilon}
\end{aligned}
$$

其中令 $f(\cdot )=\nabla_{\alpha}\mathcal{L}_{train}(\cdot,\alpha),A=\nabla_{w'}\mathcal{L}_{val}(w',\alpha),x_0=w,w^{\pm}=w\pm\epsilon\nabla_{w^{\prime}}\mathcal{L}_{val}(w^{\prime},\alpha)$，带入得：
$$
\begin{aligned}
\nabla_{\alpha,w}^2\mathcal{L}_{train}(w,\alpha)\cdot \nabla_{w'}\mathcal{L}_{val}(w',\alpha)&\approx \frac{\nabla_{\alpha}\mathcal{L}_{train}(w+\epsilon\cdot \nabla_{w'}\mathcal{L}_{val}(w',\alpha),\alpha)-\nabla_{\alpha}\mathcal{L}_{train}(w-\epsilon\cdot \nabla_{w'}\mathcal{L}_{val}(w',\alpha),\alpha)}{2\epsilon} \\
&=\frac{\nabla_{\alpha}\mathcal{L}_{train}(w^{+},\alpha)-\nabla_{\alpha}\mathcal{L}_{train}(w^{-},\alpha)}{2\epsilon}
\end{aligned}
$$

又通过一次近似，将二阶导转化为了一阶导，由此避免了二阶导的昂贵计算代价。

不过后人的实验说明，一阶导（$即\xi=0时,w^{*}(\alpha)\approx w$）的结果和二阶导（$即\xi>0时,w^{*}(\alpha)\approx w-\xi\nabla_{w}\mathcal{L}_{train}(w,\alpha)$）的结果（精度）相差不大，$\xi>0$时反而计算代价较大。

## 讨论
$\mathrm{Q}$：论文不通过完整训练，而通过一阶和二阶近似来评估权重$w$在架构$\alpha$上的表现，从而大幅降低了计算代价，那么这样是合理的吗？
$\mathrm{A}$：论文中所有操作的$w$是共用的，而不是对于第$i$次循环随机指定一个$w_i$，然后随随便便就对这个模型进行评估了，反之，架构$\alpha$和权重$w$是同时逐渐更新的，虽然$w$对于架构$\alpha$极可能不是最优的，但也是可以拿来比较评价的了。

# 问题和缺点
基于DARTS本身设计的特点，当出现两个操作权重相近时，如 $\alpha_{o_0}^{(i,j)}=0.33$ 、 $\alpha_{o_1}^{(i,j)}=0.34$ ，很难说 $\alpha_{o_0}^{(i,j)}$ 比 $\alpha_{o_1}^{(i,j)}$ 差很多，然而边 $(i,j)$ 上操作 $o_0$ 将被丢弃，这是不太合理的。


















