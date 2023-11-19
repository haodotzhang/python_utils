--
-- 长视频对TS影响比中视频大，中视频对VV影响比长视频大。AUC和GAUC都缺少对模型分品类排序情况的评估。
-- 所以，引入2组新的离线指标来评估模型对不同品类的排序情况：
-- 分品类的正样本覆盖率（Categorized Coverage，CC），分品类的TS覆盖率（Categorized Time-Spent Coverage，CTSC）
-- 这2个指标值越大模型效果越好.
-- 这个指标的定义很有趣的一点：充分利用了标签的含义。
-- 因为现在是在测试集上验证模型性能，
-- 而测试集提供了 label标签即哪些doc是真正即将会被点击的，
-- 即正样本(可以认为这些样本应该要被模型排到最前面去的，假设是M个，它们应该被排到top-M pos)。
-- 理想情况是这M个正样本被模型排到top-M位置。
-- 但实际情况下, 模型【实际】返回的top-M中可能只覆盖了 <M 个正样本 (理想情况下应该覆盖 =M 个正样本】 
-- 即Top-M的正样本覆盖率，即严格的recall (和NDCG的计算过程非常相似)
-- 同理可计算 TS_Coverage，思路也完全相同。 不是看正样本覆盖率，而是看
-- Top-M中 【正样本的sum_ts】占 全体样本Top-N中 【正样本的sum_ts】

--【回顾下NDCG的计算过程】
-- 1）根据模型预测分数倒排，返回top-k 
-- 2）依据每个doc在原始数据集中的标注分数计算DCG 
-- 3）对上述top-k doc list 基于它们在在原始数据集中的标注分数 重排，再计算DCG 即 iDCG - NDCG = DCG / iDCG

-- 小tips: 离线虽然只有一个实验桶，但有多个model_version；线上有多个实验桶，但每个实验桶只有一个model_version。
-- 本质上都是 看 分组实验结果。【离线基于 model_version模型版本号，线上基于bts分桶号】
-- 所以 SQL code 的主体计算逻辑 可以 共享。
-- 打分记录准备部分逻辑 离在线各自有不同的写法细节处理，不共享

@logs:= SELECT 
            aaid, utdid, query, eid, id,
            -- bts_rank AS bts,
            bts_rank_inner3 AS bts,
            rs_rank_model_score AS prediction
        FROM proj_name.pvlog_name
        WHERE ds = '${bizdate}'                             -- 看昨天大QD打分情况的离线数据 (今天未结束，还没数据)
        AND aaid IS NOT NULL
        AND utdid IS NOT NULL
        --AND bts_rank NOT IN (516, 504, 517)                    -- rank层
        AND bts_rank_inner3 in (801, 802, 803, 804, 805, 807, 810)                 -- rank_inner3层
        ;

@samples := SELECT 
                aaid, utdid, query, eid, id, ts,
                IF(num_vv > 0, 1, 0) AS label
            FROM proj_name.ugc_session_doc_action_filterd_d  -- 线上真实分布 
            WHERE ds = '${bizdate}'
            AND card IN ('Feed区', '短视频模块')
            ;

@inferences :=  SELECT a.bts, a.aaid, a.query, a.id, a.prediction, b.ts, b.label, a.bts AS version
                FROM @logs a INNER JOIN @samples b
                ON a.aaid = b.aaid AND a.utdid = b.utdid AND a.eid = b.eid AND  a.id = b.id
                ;

@ranks  :=  SELECT 
                version, query, id, ts, 
                ROW_NUMBER() OVER (PARTITION BY version, query ORDER BY prediction DESC) AS rk   -- 用 query分组 代替 用户分组 （计算分query下的 TSC）；组内排序
            FROM @inferences
            ;

@thres  :=  SELECT version, query, SUM(label) AS threshold   -- 计算每个组内，每个query下的正样本个数 作为公式中的 C
            FROM @inferences 
            GROUP BY version, query
            ;

@scores :=  SELECT 
                version, query,
                SUM(IF(rk <= threshold, ts, 0)) AS valid_ts,  -- 计算每个组内，每个query下 C个正样本 被覆盖了多少个？【并且是增强版recall: 看top-C中有多少个 正样本 而不是看top-K】; 累积的不是个数 是TS
                SUM(ts) AS ts                                 -- 只有正样本有ts，所以忽略负样本 直接 sum(all ts)
            FROM (
                SELECT 
                    a.version AS version, 
                    a.query AS query,
                    rk, ts, threshold
                FROM @ranks a JOIN @thres b
                ON a.version = b.version AND a.query = b.query
            ) GROUP BY version, query
            ;

@tsc:=  SELECT 
            version, 
            SUM(valid_ts) / SUM(ts) AS tsc 
        FROM @scores GROUP BY version;

-- INSERT OVERWRITE TABLE ads_ugc_models_tsc_d PARTITION (ds = '${bizdate}', version)
-- SELECT tsc, version FROM @gauc ORDER BY gauc DESC;

SELECT '${bizdate}' AS ds, * FROM @tsc ORDER BY tsc DESC;