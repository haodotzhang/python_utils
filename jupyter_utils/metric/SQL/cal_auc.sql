
@logs:= SELECT 
            aaid, utdid, query, eid, id,
            -- bts_rank AS bts,
            bts_rank_inner3 AS bts,
            rs_rank_model_score AS prediction
        FROM proj_name.pvlog_name
        WHERE ds = '${bizdate}'                             -- 看昨天大QD打分情况的离线数据 (今天未结束，还没数据)
        AND aaid IS NOT NULL
        AND utdid IS NOT NULL
        -- AND bts_rank NOT IN (, 504, 517) -- rank层
        AND bts_rank_inner3 in (801, 802, 803, 804, 805, 807, 810)                 -- rank_inner3层
        ;

-- 过滤query
-- @target_query :=    SELECT query 
--                     FROM proj_name.ads_query_search_stats_feature_d
--                     WHERE ds = '${bizdate}'                 -- 昨天大QD打分所用的低频query 来自 昨天低频表
--                     AND tot_exp_pv_1 <= 10
--                     ;

@samples := SELECT 
                aaid, utdid, query, eid, id,
                IF(num_vv > 0, 1, 0) AS label
            FROM proj_name.ads_ugc_session_doc_action_filterd_d  -- 线上真实分布 
            WHERE ds = '${bizdate}'
            AND card IN ('Feed区', '短视频模块')
            --AND query IN (SELECT * FROM @target_query)
            ;

@inferences :=  SELECT a.bts, a.aaid, a.query, a.prediction, b.label
                FROM @logs a INNER JOIN @samples b
                ON a.aaid = b.aaid AND a.utdid = b.utdid AND a.eid = b.eid AND a.id = b.id
                ;

@ranks :=   SELECT 
                bts, prediction, (MAX(id) + MIN(id)) / 2.0 AS frank
            FROM (
                SELECT bts, prediction, ROW_NUMBER() OVER (PARTITION BY bts ORDER BY prediction ASC) AS id
                FROM @inferences
            ) GROUP BY bts, prediction
            -- GROUP BY bts, prediction的目的： 保证 同一天内同一个桶bts内 相等prediction 多个打分记录 的排序位置相同 (具体做法是 [max_id+ min_id] / 2)
            -- 相等prediction的多个打分记录：在一天内同一个bts内 某个Q-D对 由于不同打分请求产生的多次被打分。除非有实时特征prediction可能不同，否则应该每次都相同)  【因为模型特征都没变化，同一个Q-D的模型打分应该相同】
            -- 因为现在无实时特征，理论上不存在 【在一天内同一个bts内 某个Q-D对 由于不同打分请求产生的多次被打分的prediction 不同】，否则日志系统或者 线上的特征/模型链路有bug
            ;

@auc := SELECT
            bts,
            (
                (SUM(label*frank) - SUM(label) * (SUM(label) + 1) / 2.0) / 
                (SUM(label) * (COUNT(*) - SUM(label)))
            ) AS auc
        FROM (
            SELECT a.bts, a.prediction, frank, label
            FROM @inferences a JOIN @ranks b
            ON a.bts = b.bts AND a.prediction = b.prediction
        ) GROUP BY bts
        ;

-- INSERT OVERWRITE TABLE ads_ugc_model_online_auc_d PARTITION (ds = '${bizdate}')
-- SELECT auc FROM @auc;

SELECT '${bizdate}' as ds, * FROM @auc;