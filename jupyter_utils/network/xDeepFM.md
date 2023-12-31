- 【优化点的本质】xdeepfm vs DCN
 - diss了DCN中所谓的高阶交叉本质是X0乘以一个常数，并没有真正的高阶交叉
 - 引入CIN实现任意阶的vector_wise的特征交叉，输入特征/中间层feature map，及权重参数 都保持为一个矩阵而非向量的形式

- 【可视化】不同网络权重层，学到了什么东西？比如CIN学到了什么交叉信息？如何验证？比如DIN学习到了什么信息？如何验证？
X^0是输入特征矩阵，是mxD，即m个特征，每个是D-dim vector。
CIN的第一层的W^1，可以可视化 X^0和X^0之间的二阶交叉<X_i, X_j>，具体来说是每一个feature与其他feature之间的权重。越大越相关。
「所谓几阶交叉，指的是几个特征组为group建模相关性, 比如<F1,F2>即为二阶，<F1,F2,F3>即为3阶，<F1,F2,....Fn>即为n阶」
再往后的高阶交叉，可视化的物理意义就不可解释了。


DNN其实是MLP的输入X是一个拉长的D-dim vector，已经损失了feature_wise的意义，所以DNN建模的是 隐式的不可控不可解释的bit_wise交叉


注意CIN中，
1) 一个W^{k}的计算过程如下： (它的shape为 H_{k-1} x m，m是X^0是输入特征个数，H_{k-1}是上一次输出也即本层输入特征feature map的行数，也是特征个数。待会儿会产生 H_{k-1} x m 二阶组合，每个二阶组合都会对 两输入向量执行hadmatdot product，输出一个同维的D-dim向量)
X^{k-1} _{i} 和 X^{0} _{j} 这两个向量首先做的是 hadmatdot product，返回是一个与输入同维的向量。然后再施加一个W_ij的常数权重，对向量做长度上的伸缩。然后 sigma_{i,j}，即sum_pooling为一个D-im vector
2) 第k层有H_{k}个W^{k}，然后将H_{k}个 根据上面公式计算得到的D-im vector 拼接为一个 X^{k}，shape是 H_{k} x D。这个过程非常像CNN的卷积操作
所以第k层，权重参数W的个数是 H_{k} x H_{k-1} x m。这个参数计算和CNN卷积的计算过程很像。
所以总共参数量为: L x H_{k} x H_{k-1} x m (这里假设每一层参数量接近，为了表达方便),其实实际上是 sigma_{k} (H_{k} x H_{k-1} x m)

3) 并且1）中计算过程，当前层X^{k}依赖上一层X^{k-1}和X^{0}，计算公式形式很像RNN(依赖当前上一层和当前输入)。只不过这里的当前输出是恒定是X^{0}

4）上述1）和2）的理论计算过程 也可以等价理解为一个 特殊的卷积过程：【所以代码实践层次，就可以借助CUDA加速】
在第k层，
输入Z^{k} = <X^{k-1}, X^{0}>的外积 = 一个 H(k-1) x m x D 是3维tensor，这里做完了一个外积运算
一个特殊的卷积核W^{k} 是一个 H(k-1) x m x 1 是3维tensor，
W^{k} 会逐个通道和Z^{k} 做卷积(两个H(k-1) x m的矩阵 做一次卷积的结果是 1个常数 )，得到一个常数。然后D个通道就得到D个常数，拼接为一个D-dim vector
至此，可以看到这个卷积核的特殊之处：首先是kernel_size和input_size 在长宽方向 完全相同；其次D个通道共享一个卷积核参数。

上面展示了第k层 1个卷积核和Z^{k} 昨晚卷积后的结果，一个D-dim vector。而第k层有H_{k}个卷积核，所以最终输出的X^{k}是 H_{k} x D，即H_{k}个D-dim vector。回顾一下输入输出和卷积核：输入H_{k-1} x m x D, 输出H_{k} x D，卷积核 H_{k} x H_{k-1} x m x 1。中间卷积过程的参与对象是 两个H_{k-1} x m 的矩阵 ，逐通道执行，再将D个通道的结果拼接为一个D-dim vector。

5）获得每一层X^{k} = H_{k} x D 的输出特征feature map后，如何得到最终X^{T}的输出呢？
其实第k层的输出 X^{k} = H_{k} x D 中有 H_{k} 个 D-dim vector，每个D-dim vector 都蕴含了1~k+1阶的特征交叉(交叉信息即特征相关性 存储在W^{k}中)，所以一个简单做法是 做sum_pooling，将 X^{k} = H_{k} x D 降解为一个 H_{k}维向量。即第k层最终的输出是一个H_{k}维向量。
最终，拼接1~T层的多个输出向量为一个向量，<H_{1}, H_{2}... H_{T}>，维度是 sigma_{k}^H_{k}，可近似为 T * H_{k}(这里假设每一层的卷积核个数相同 for 分析方便)

6）CIN层输出一个向量后，再接一个MLP输出层转化为1-dim logits。即为CIN层最终的logits输出注意CIN中，
1) 一个W^{k}的计算过程如下： (它的shape为 H_{k-1} x m，m是X^0是输入特征个数，H_{k-1}是上一次输出也即本层输入特征feature map的行数，也是特征个数。待会儿会产生 H_{k-1} x m 二阶组合，每个二阶组合都会对 两输入向量执行hadmatdot product，输出一个同维的D-dim向量)
X^{k-1} _{i} 和 X^{0} _{j} 这两个向量首先做的是 hadmatdot product，返回是一个与输入同维的向量。然后再施加一个W_ij的常数权重，对向量做长度上的伸缩。然后 sigma_{i,j}，即sum_pooling为一个D-im vector
2) 第k层有H_{k}个W^{k}，然后将H_{k}个 根据上面公式计算得到的D-im vector 拼接为一个 X^{k}，shape是 H_{k} x D。这个过程非常像CNN的卷积操作
所以第k层，权重参数W的个数是 H_{k} x H_{k-1} x m。这个参数计算和CNN卷积的计算过程很像。
所以总共参数量为: L x H_{k} x H_{k-1} x m (这里假设每一层参数量接近，为了表达方便),其实实际上是 sigma_{k} (H_{k} x H_{k-1} x m)

3) 并且1）中计算过程，当前层X^{k}依赖上一层X^{k-1}和X^{0}，计算公式形式很像RNN(依赖当前上一层和当前输入)。只不过这里的当前输出是恒定是X^{0}

4）上述1）和2）的理论计算过程 也可以等价理解为一个 特殊的卷积过程：【所以代码实践层次，就可以借助CUDA加速】
在第k层，
输入Z^{k} = <X^{k-1}, X^{0}>的外积 = 一个 H(k-1) x m x D 是3维tensor，这里做完了一个外积运算
一个特殊的卷积核W^{k} 是一个 H(k-1) x m x 1 是3维tensor，
W^{k} 会逐个通道和Z^{k} 做卷积(两个H(k-1) x m的矩阵 做一次卷积的结果是 1个常数 )，得到一个常数。然后D个通道就得到D个常数，拼接为一个D-dim vector
至此，可以看到这个卷积核的特殊之处：首先是kernel_size和input_size 在长宽方向 完全相同；其次D个通道共享一个卷积核参数。

上面展示了第k层 1个卷积核和Z^{k} 昨晚卷积后的结果，一个D-dim vector。而第k层有H_{k}个卷积核，所以最终输出的X^{k}是 H_{k} x D，即H_{k}个D-dim vector。回顾一下输入输出和卷积核：输入H_{k-1} x m x D, 输出H_{k} x D，卷积核 H_{k} x H_{k-1} x m x 1。中间卷积过程的参与对象是 两个H_{k-1} x m 的矩阵 ，逐通道执行，再将D个通道的结果拼接为一个D-dim vector。

5）获得每一层X^{k} = H_{k} x D 的输出特征feature map后，如何得到最终X^{T}的输出呢？
其实第k层的输出 X^{k} = H_{k} x D 中有 H_{k} 个 D-dim vector，每个D-dim vector 都蕴含了1~k+1阶的特征交叉(交叉信息即特征相关性 存储在W^{k}中)，所以一个简单做法是 做sum_pooling，将 X^{k} = H_{k} x D 降解为一个 H_{k}维向量。即第k层最终的输出是一个H_{k}维向量。
最终，拼接1~T层的多个输出向量为一个向量，<H_{1}, H_{2}... H_{T}>，维度是 sigma_{k}^H_{k}，可近似为 T * H_{k}(这里假设每一层的卷积核个数相同 for 分析方便)

6）CIN层输出一个向量后，再接一个MLP输出层转化为1-dim logits。即为CIN层最终的logits输出


