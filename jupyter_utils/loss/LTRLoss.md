CPP Loss 
1）celoss在tf中的实践：
- https://paddlepedia.readthedocs.io/en/latest/tutorials/deep_learning/loss_functions/CE_Loss.html
- https://www.tensorflow.org/api_docs/python/tf/compat/v1/losses/sigmoid_cross_entropy （tf最上层封装，除了celoss，还加了weights/label_smoothing/reduction）
- https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits (真正执行 sigmoid_celoss的函数，包含了全部计算细节)
【二分类】针对某个样本，有3处简化：
a) 直接用sigmoid简化代替softmax对inference的logits做归一化，
b) y_real不需要做one-hot 直接使用单值 y_real  (默认设置正类为1.0，父类为0.0)，
c) y_logits经过归一化层sigmoid的输出是单个值y_pred (默认输出的是预测为正类的概率，比如0.6)，
对应的简化公式为：celoss(y_real, y_pred) = - [y_real * log(y_pred) +  (1-y_real) * log(1-y_pred)]，这个公式展开覆盖了两类情况，用y_real作为分类讨论的属性
- 如果是正样本，则y_real=1.0，真正使用的是第一项，正确预测为正类的损失 l+ = -log(y_pred)，loss bp会鼓励y_pred越大越好；第二项为0，不考虑它把正样本误分类成负类的损失。
- 如果是负样本，则y_real=0.0，真正使用的是第二项，正确预测为负类的概率是(1-y_pred)，而损失 l- = - log(1-y_pred)，loss bp会鼓励(1-y_pred)越大越好。也即y_pred越小越好。第一项为0，即不考虑把它错误预测为正类的loss。
综上，对于每一个样本，celoss 只计算 【当前样本被正确预测为其real_label】的loss，鼓励这个p_pred越大越好

【多分类】
a) sigmoid-> softmax, 对inference的logits做归一化，
b) y_real做one-hot，(1.0, 0.0) 或者 (0.0, 1.0)
c) y_logits经过归一化层sigmoid的输出y_pred = (y1, y2)，比如 (0.6, 0.4) 或 (0.3, 0.7)
celoss(y_real, y_pred) = - y_real * log(y_pred)，其实是包含了y_real在不同取值下的不同情况的计算过程：
- 如果是正样本，则y_real=(1.0, 0.0)，则celoss= - log(y1)，其实只需要计算在y_real=1 bit上(即y1项)的运算，其余bit位上因为y_real为0，结果均为0。
- 如果是负样本，则y_real=(0.0, 1.0)，则celoss= - log(y2)，同样只需要计算在y_real=1 bit上的运算(即y2项)
- 扩展到N分类，同样：在每一种y_real=yk的取值情况下，只需要计算 yk 下的结果 -log(yk)
综上，对于每一个样本，celoss 只计算 【当前样本被正确预测为其real_label k-th】的loss，鼓励这个p_pred越大越好，即鼓励 y_pred 单峰分布

2）loss func
def pairwise_cross_entropy_v2(
    session_id, labels, logits,
    weights=1.0, use_weight=False,
    margin=10.0, use_margin=True,
    add_batchin_neg=False, sample_mask=None, ts=None):
    if sample_mask is not None:
        session_id = tf.reshape(tf.boolean_mask(session_id, sample_mask), [-1, 1])
        labels = tf.reshape(tf.boolean_mask(labels, sample_mask), [-1, 1])
        logits = tf.reshape(tf.boolean_mask(logits, sample_mask), [-1, 1])
        weights = tf.reshape(tf.boolean_mask(weights, sample_mask), [-1, 1])

    session_mask = tf.cast(tf.equal(session_id, tf.transpose(session_id)), tf.float32)
    session_mask = session_mask - tf.matrix_diag(tf.diag_part(session_mask))

    if add_batchin_neg:
        session_outer_mask = tf.transpose(tf.random_shuffle(tf.transpose(session_mask)))
        session_mask = session_mask + session_outer_mask
        session_mask = session_mask - tf.matrix_diag(tf.diag_part(session_mask))

    labels_mask = tf.greater(labels - tf.transpose(labels), 0)
    ts_mask = tf.greater(ts - tf.transpose(ts), 0)
    labels_mask = tf.logical_or(labels_mask, ts_mask) 
    labels_mask = tf.cast(labels_mask, tf.float32)
    final_mask = tf.multiply(session_mask, labels_mask)

    logits_mat = logits - tf.transpose(logits)
    if use_margin:
        logits_mask = tf.cast(tf.less(logits_mat, margin), tf.float32)
        final_mask = tf.multiply(final_mask, logits_mask)

    final_mask = tf.cast(final_mask, tf.bool)
    logits_mat_valid = tf.boolean_mask(logits_mat, final_mask)

    if use_weight:
        weights_mat = tf.abs(weights - tf.transpose(weights))
        weights = tf.boolean_mask(weights_mat, final_mask)
    else:
        weights = tf.ones_like(logits_mat_valid)

    labels_mat_psudo = tf.ones_like(logits_mat_valid)

    loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=labels_mat_psudo, logits=logits_mat_valid, weights=weights)
    loss = tf.where(tf.is_nan(loss), tf.zeros_like(loss), loss)

    return loss

3) def compute_loss():
self.losses[task] = self.get_loss_hyperparameter("point_weight") * globals()[self.get_loss_hyperparameter("point_loss")](task_labels, task_logits, weights=weights, sample_mask=sample_mask) + \
                                            self.get_loss_hyperparameter("pair_weight") * globals()[self.get_loss_hyperparameter("pair_loss")](session_id, task_labels, task_logits, weights=pair_ind_weights, sample_mask=sample_mask, 
                                                                                                                                                ts=ts,
                                                                                                                                                **self.get_loss_hyperparameter("pairwise"))   
                    
4）def build_graph():
self.loss = self.compute_loss(self.outputs, self.labels, self.weights, session_id, sample_mask, self.pair_ind_weights, ts)



实践写代码的经验：
- 多样本的二分类在写代码实现celoss时，可直接使用 单个样本的多分类的函数 sigmoid_cross_entropy()，因为推导过程相同。见上面的code。以及https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits 的示例。
- 实际代码计算的加速技巧：
- sigmoid/softmax 与 cross_entropy_loss 公式结合后，得到的 sigmod_cross_entropy_loss(x) 对 x 求导，求导过程相比拆成两步，大大简化，简化了计算操作，加速训练。(见https://paddlepedia.readthedocs.io/en/latest/tutorials/deep_learning/loss_functions/CE_Loss.html。本质是算子合并，加速训练)
- 在实际输入给 sigmod_cross_entropy()的 input x 中，过滤掉y_real中为0的bit，减少了不必要的计算，加速训练

实践代码优化：
max(x, 0) - x * z + log(1 + exp(-abs(x)))，包含了两种情况
- x > 0时，x - x * z + log(1 + exp(-x))
- x <0时，- x * z + log(1 + exp(x))，避免 exp(-x)太大导致的计算溢出


cpp loss
1) 样本侧：
- 离线训练集要将 同一用户同一请求(utdid+aaid)聚集在一起
- 正负构成pair，正-正样本也构成pair, 并且限制只有 同一用户同一请求内的样本构成pair
- 过滤某些请求的pair，比如无曝光session
- 对 样本pair 添加 VV/TS加权

2) Loss侧
- celoss 换成 hinge_loss? 实践上的意义不大

分析：
- 在模型训练时，为了让相同用户的样本尽可能多地在一个Mini-batch内构成Pair来训练，对样本按照用户ID排序。
- 并且，因为时长是视频推荐领域的关键指标，为了建模时长信息，使用2个样本的TS差值的绝对值来为Pair加权
- 在Loss设计中使在同一Session中的Pair比在不同Session间的Pair权重更高，因为同一个Session内的样本的下滑深度和时间等环境Bias几乎一致，所以在同一个Session内构成的Pair比在不同Session间构成的Pair更有学习价值
- 为了避免没有消费的用户不会被模型学习的情况，在Loss后添加交叉熵。「使用CPP而不是简单的 pairwise loss取代pointwise loss」


枫少的Weighted Pairwise Loss
def weighted_gauc_loss(utdid_tensor, ts_tensor, label_tensor, logit_tensor, use_weight=False, alpha=1.0,
                       epsilon=1E-6):
    utdid_mask = tf.cast(tf.equal(utdid_tensor, tf.transpose(utdid_tensor)), tf.float32)
    utdid_mask = utdid_mask - tf.matrix_diag(tf.diag_part(utdid_mask))

    label_mask = 1 - tf.cast(tf.equal(label_tensor, tf.transpose(label_tensor)), tf.float32)
    label_mask = label_mask - tf.matrix_diag(tf.diag_part(label_mask))

    final_mask = tf.multiply(utdid_mask, label_mask)

    prob_mat = tf.sigmoid(logit_tensor - tf.transpose(logit_tensor))
    ts_mat = alpha * tf.abs(ts_tensor - tf.transpose(ts_tensor)) + 1.0
    label_mat = (label_tensor - tf.transpose(label_tensor) + 1.0) / 2.0

    final_weight = final_mask

    ent_mat = -tf.multiply(label_mat, tf.log(prob_mat + epsilon)) - tf.multiply(
        (1 - label_mat), tf.log(1 - prob_mat + epsilon)) # 手写 二分类的celoss
    if use_weight:
        final_weight = tf.multiply(final_mask, ts_mat)

    return tf.reduce_sum(tf.multiply(ent_mat, final_weight)) / (tf.reduce_sum(final_mask) + epsilon) # 对多个样本的celoss做reduce


    淘主的CPP Loss
pairwise loss优化动机
- 引入ranking loss基于这样一个简单的动机，“让模型训练的优化目标和在/离线评价指标尽量保持一致”。「大部分排序模型的离线评价指标是AUC/GAUC(任意正样本比负样本打分高的概率), 但是模型的优化目标是分类任务的交叉熵，这样明显的不一致可能会导致模型优化效率降低(需要更多数据来达到同样的性能)」
- 通过pairwise ranking loss来直接优化AUC, 同时通过交叉熵来保持输出分一定的准度。
- pairwise的重要意义在于让模型的训练目标和模型实际的任务之间尽量统一。对于一个排序任务，真实的目标是让正样本的预估分数比负样本的高，对应了AUC这样的指标。pairwise loss也被称为AUC loss。
- 如何构造pair？需要根据场景特点选择。如果选择in-batch random pair，那优化pairwise ranking loss对应的离线指标就是total AUC。如果pair的构建方式的同user, 或者同user + query的，那ranking loss对应的离线指标是user AUC或者user+query AUC。


淘主 listwise loss (listwise建模1：loss侧建模list间序关系)
- 优化一个list里面正负样本的softmax loss，就是在优化正样本排在top1的概率。所谓softmax loss，即将二分类的 sigmoid_celoss 换成 softmax_celoss，即 -sigma {yi * log(softmax(yi^))}。依据这个loss coding
- list样本组织：就延续用 pairwise loss的样本组织


def listwise_softmax_cross_entropy3(session_id, labels, logits, weights=None, sample_mask=None, logits_weight=1.0):
    _EPSILON = 1e-10
    if sample_mask is not None:
        session_id = tf.reshape(tf.boolean_mask(session_id, sample_mask), [-1, 1])
        labels = tf.reshape(tf.boolean_mask(labels, sample_mask), [-1, 1])
        logits = tf.reshape(tf.boolean_mask(logits, sample_mask), [-1, 1])
        if weights is not None:
            weights = tf.reshape(tf.boolean_mask(weights, sample_mask), [-1, 1])

    session_mask = tf.equal(session_id, tf.transpose(session_id))
    session_mask_f = tf.cast(session_mask, tf.float32)

    logits = tf.reshape(logits, [-1, 1])
    logits_bb = logits - tf.zeros_like(tf.transpose(logits))
    logits_final = tf.where(session_mask, logits_bb, tf.log(_EPSILON) * tf.ones_like(logits_bb))
    
    labels = tf.reshape(labels, [-1, 1])
    labels_bb = labels - tf.zeros_like(tf.transpose(labels))
    labels_final = tf.where(session_mask, labels_bb, tf.zeros_like(labels_bb))

    if weights is not None:
        weights = tf.reshape(weights, [-1, 1])
        weights_bb = weights - tf.zeros_like(tf.transpose(weights))
        weights_final = tf.where(session_mask, weights_bb, tf.zeros_like(weights_bb))
    else:
        weights_final = tf.ones_like(labels_final)

    label_sum = tf.reduce_sum(input_tensor=labels_final, axis=0, keep_dims=True)
    nonzero_mask = tf.greater(tf.reshape(label_sum, 
[-1]
), 0.0)
    padded_labels = tf.where(nonzero_mask, labels_final, _EPSILON * tf.ones_like(labels_final)) * session_mask_f
    padded_label_sum = tf.reduce_sum(input_tensor=padded_labels, axis=0, keep_dims=True)
    normalized_labels = padded_labels / padded_label_sum

    exps = tf.exp(logits_weight * logits_final) * session_mask_f
    softmax = tf.div_no_nan(exps, tf.reduce_sum(exps, axis=0))
    losses = -tf.reduce_sum(normalized_labels * tf.log(softmax + _EPSILON) * weights_final * session_mask_f, axis=0)

    per_row_weights = tf.reduce_sum(session_mask_f, axis=1)
    session_cnt = tf.reduce_sum(1.0 / per_row_weights)
    listwise_loss = tf.reduce_sum(losses / per_row_weights) / session_cnt

    return listwise_loss



    淘主 listwise loss (listwise建模2：model侧建模list间序关系)
- 样本：list in, list out, 从mini-batch中截断30个同session-同user的doc list，不足的mini-batch就丢掉
- 特征：doc的一系列特征，和其他正常模型一致。
- model: transformer: input(30, D), output(30, 1)。输入doc特征，输出每个doc的y_pred_1。(y_pred_2是正常的非trasnformer的输出)
- loss: 正常的 sigmoid_celoss(y_real, y_pred_1+y_pred_1)

当前简化处理了：训练和测试时，输入的input_size完全相同，都是30个doc作为输入喂给模型。其实不需要一致



           # Concat mlp_out and cin_out. （就是正常的(30, D) , 拼接MLP和CIN的输出）
            final_layer = tf.concat([mlp_out, cin_out], axis=-1)
            
           # 下面操作非常简单：final_layer经过一层FC获得 y_pred_2，并行通过一个transformer 获得 y_pred_2 (比我想象的 利用tranformer的做法，要简单很多)

            deep_logit = tf.contrib.layers.fully_connected(final_layer, 1, activation_fn=None, biases_initializer=None)

            # Transformer.
            transf_out = self.transformer(final_layer, is_need_mask)
            transf_logit = tf.contrib.layers.fully_connected(transf_out, 1, activation_fn=None, biases_initializer=None)

            self.logits = deep_logit + transf_logit


https://www.mdnice.com/writing/d054bb513adb4dbf9d4aad9946c7cd66

pairwise loss优化有个待优化点：label侧配好pair label并计算pair loss，同时也必须要求 input侧配好 pair
「输入侧最好包含 pair信号，哪怕是简单的id-id」，输出pair的比较logits才有意义。
这样要求：不仅改loss，还要改样本组织，还要改模型结构「支持输入pair信号，比如 id-id」

Q：重排侧为啥没显式考虑？A：重排侧pairwise loss优化是结合了 transformer网络结构优化，本来是30 in, 30 out，天然就包含了 2-in, 2-out
启发：精排侧的模型结构和样本输入，还得优化才能真正用对。
- 记得处理边界：当前inferance的input batch_szie至少是2的倍数，多余的丢掉」
- 关于pair sample，训练时需要输入，测试时可以不输入，因为测试时并不需要pair preference，只需要输出point ctr (debias也是测试时不需要输入pos)
- 甚至可以构造，query-pair_id 特征


上述loss思路来自下面的论文 lisnet:
ListNet [8] Learning to Rank: From Pairwise Approach to Listwise Approach, ICML '07, MSRA

排序学习-3.排序学习模型
Learning to Rank学习笔记--ListwiseRank
Learning to Rank : ListNet与ListMLE_DS..的博客-CSDN博客 (截至目前最为清晰的一个解读，关于listnet和listmle；而且描述数学公式和讲解非常清晰，学习)

【有空再仔细思考它的理论及TF代码实践，尤其是ATA中用到的代码】
代码分析：TODO