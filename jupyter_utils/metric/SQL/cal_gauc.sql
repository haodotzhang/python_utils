-- 写法1
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
                aaid, utdid, query, eid, id,
                IF(num_vv > 0, 1, 0) AS label
            FROM proj_name.ads_ugc_session_doc_action_filterd_d  -- 线上真实分布 
            WHERE ds = '${bizdate}'
            AND card IN ('Feed区', '短视频模块')
            ;

@inferences :=  SELECT a.bts, a.aaid, a.query, a.prediction, b.label
                FROM @logs a INNER JOIN @samples b
                ON a.aaid = b.aaid AND a.utdid = b.utdid AND a.eid = b.eid AND a.id = b.id
                ;

@ranks:=SELECT 
            bts, query, prediction, (MAX(id) + MIN(id)) / 2.0 AS frank
        FROM (
            SELECT 
                bts, query, prediction, 
                ROW_NUMBER() OVER (PARTITION BY bts, query ORDER BY prediction ASC) AS id
            FROM @inferences
        ) GROUP BY bts, query, prediction
        ;

@auc := SELECT
            bts, query,
            (
                (SUM(label*frank) - SUM(label) * (SUM(label) + 1) / 2.0) / 
                (SUM(label) * (COUNT(*) - SUM(label)))
            ) AS auc
        FROM (
            SELECT a.bts, a.query, a.prediction, frank, label
            FROM @inferences a JOIN @ranks b
            ON a.bts = b.bts AND a.query = b.query AND a.prediction = b.prediction
        ) GROUP BY bts, query -- 先基于bts分组，在同一个bts内再基于query对打分记录分组 (计算同一个bts同一个query为一组的 组内AUC)
        ;

@pv :=  SELECT 
            bts, query,
            COUNT(DISTINCT aaid) AS pv
        FROM @inferences
        GROUP BY bts, query -- 基于bts, query对打分记录分组，计算同一个bts桶内 每个query的pv
        ;

@tpv := SELECT 
            bts, SUM(pv) AS total_pv
        FROM @pv
        GROUP BY bts -- 计算 同一个桶内 全部query的总pv
        ;

-- 因为此处total_pv与具体query无关，是一个定值，能提前计算出来，可以在这里 把 bts, query, pv, total_pv, weight=pv/total_pv 拼接为一行 (简化gauc的公式)
-- 下面直接 SUM(weight * auc) AS gauc

-- 并且也支持看 不同query分组内的 AUC 以便分析 降效query



-------------------------------------------------------------------------------------------------------
-- 写法2
SELECT bts, query, pv, total_pv, pv/total_pv AS weight 
FROM (
    (
        SELECT bts, query, pv
        FROM @pv
    ) a
    INNER JOIN 
    (
        SELECT bts, total_pv
        FROM @tpv
    ) b 
    ON a.bts = b.bts
)

@gauc:= SELECT 
            a.bts, 
            gauc / total_pv AS gauc
        FROM (
            SELECT 
                a.bts, 
                SUM(pv * auc) AS gauc   --本质是想做加权求和，只是暂时用的是pv，后面需要再除以总pv, 才是真正的权重 pv/total_pv
            FROM @auc a 
            INNER JOIN @pv b
            ON a.bts = b.bts AND a.query = b.query
            GROUP BY a.bts
        ) a INNER JOIN @tpv c
        ON a.bts = c.bts
        ;

SELECT '${bizdate}' as ds, * FROM @gauc ORDER BY gauc DESC;