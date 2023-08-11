from TextClean import TextProcessor
from FileHelper import CSVHelper
import os, csv, re 
import pandas as pd


# 基于csv 很方便自定义 csv_load 和 csv_write TODO 
def load_csv():
    pass 

def write_csv():
    pass

def load_txt():
    pass 

def write_txt():
    pass


class PandasDataset:
    pass 


'''文本清洗'''
def clean_text_v1(text_pro, text):
    # path_stopwords = './lib_resource/stop_words.txt'
    # text_pro = TextProcessor(path_stopwords=path_stopwords)
    ret_list = text_pro.text_preprocess_v2(text)
    # print("ret_list: ", ret_list)
    # print("ret_text: ", ','.join(ret_list))
    ret_text = ','.join(ret_list)

    return ret_text

'''生成必备文本-中间过程'''
def preprocess_baiduqa():
    path_stopwords = './lib_resource/stopwords/baidu_stopwords.txt'
    text_pro = TextProcessor(path_stopwords=path_stopwords)

    # 19w7数据：选择15w参与模型优化（保留4w7~5w做纯粹的 误报率测试集）
    path_ds = '/home/zh342219/personal_repo/Python/text_classification'
    data_path = os.path.join(path_ds, "inference_dir/BaiduQA_20230809.csv") # 
    path_neg_baiduqa_4_model = os.path.join(path_ds, "baiduqa_4_model.csv") # 4_preserve
    path_neg_baiduqa_4_preserve = os.path.join(path_ds, "baiduqa_4_preserve.csv") # 4_preserve

    # load csv
    line_list = load_csv(data_path)
    # 随机sample 分成两部分
    line_model = line_list[:150000]
    line_preserve = line_list[150000:]
    # 清洗 suammry NOTE 必做
    data_list_4_model = []
    for line in line_model:
          data = clean_text_v1(text_pro, line[-1]) # 只取最后一列summary
          # 是否需要依据长度 过滤 data ? TODO
          if len(data) > 0:
               data_list_4_model.append([data]) # 注意是 [data] 代表一行 (这一行就1列) 
    # write csv
    header = ['summary']
    write_csv(path_neg_baiduqa_4_model, header, data_list_4_model)
    header = ['qid', 'query', 'answer', 'summary']
    write_csv(path_neg_baiduqa_4_preserve, header, line_preserve)


'''生成数据集过程'''
def gen_pd_ds_v3_1(pos_input, neg_input, neg_baiduqa_input, train_output, test_output, times = 1.0):
    # 不做任何筛选，正负: 4w:15w (pos+neg_1: neg_2), 任务转化为: 是否为政治主题分类
    # v3_1_1: 正负样本比: 1:2
    # v3_1_2: 正负样本比: 1:3
    # v3_1_3: 正负样本比: 1:4 (15不足16，取15w)

    pos_text_list = load_txt(pos_input) # 2w
    neg_text_list = load_txt(neg_input) # 2w
    neg_baiduqa_list = load_csv(neg_baiduqa_input) # 共15w, 即选择了15w参与模型优化（保留了4w7~5w做纯粹的 误报率测试集）
    # 配置label,并重置为二维数组
    pos_pair = [[pos_text, 1] for pos_text in pos_text_list]
    neg_pair = [[neg_text, 1] for neg_text in neg_text_list] # NOTE 注意lable是 1
    neg_baidu_pair = [[neg_text[0], 0] for neg_text in neg_baiduqa_list] # 注意 neg_text[0]
    
    df_pos = pd.DataFrame(pos_pair, columns=['text', 'label'])
    df_neg = pd.DataFrame(neg_pair, columns=['text', 'label'])
    df_neg_baiduqa = pd.DataFrame(neg_baidu_pair, columns=['text', 'label'])
    print("len_df_pos: ", df_pos.shape[0], df_pos.shape[1])
    print("len_df_neg: ", df_neg.shape[0], df_neg.shape[1])
    print("len_df_neg_baiduqa: ", df_neg_baiduqa.shape[0], df_neg_baiduqa.shape[1])
    print("df_pos: ", df_pos.head(3))
    print("df_neg: ", df_neg.head(3))
    print("df_neg_baiduqa: ", df_neg_baiduqa.head(3))

    # 正负比例控制
    # times = ?
    #
    len_pos = df_pos.shape[0] + df_neg.shape[0] # NOTE 注意是4w 不是 2w
    #
    len_neg = min(len_pos*times, df_neg_baiduqa.shape[0]) 
    print("len_pos={}, len_neg={}".format(len_pos, len_neg))
    df_neg_baiduqa = df_neg_baiduqa.sample(n=len_neg, random_state=20230727) # NOTE 负样本只有baiduqa
    print("len_df_neg_baiduqa: ", df_neg_baiduqa.shape[0], df_neg_baiduqa.shape[1])
    print("df_neg_baiduqa: ", df_neg_baiduqa.head(3))

    # 基于采样比例构建 train and test
    # 训练集:测试集=8:2, # 前80%作为train, 后20%作为test
    df_train_pos = df_pos.sample(frac=0.8, random_state=20230727)
    df_train_neg = df_neg.sample(frac=0.8, random_state=20230727)
    df_train_neg_baiduqa = df_neg_baiduqa.sample(frac=0.8, random_state=20230727)
    print("df_train_pos: ", df_train_pos.shape[0], df_train_pos.shape[1])
    print("df_train_pos: ", df_train_pos.head(3))
    print("df_train_neg: ", df_train_neg.shape[0], df_train_neg.shape[1])
    print("df_train_neg: ", df_train_neg.head(3))
    print("df_train_neg_baiduqa: ", df_train_neg_baiduqa.shape[0], df_train_neg_baiduqa.shape[1])
    print("df_train_neg_baiduqa: ", df_train_neg_baiduqa.head(3))
    # 剩余样本进入test
    df_test_pos = df_pos.drop(index=df_train_pos.index)
    df_test_neg = df_neg.drop(index=df_train_neg.index)
    df_test_neg_baiduqa = df_neg_baiduqa.drop(index=df_train_neg_baiduqa.index)
    print("df_test_pos: ", df_test_pos.shape[0], df_test_pos.shape[1])
    print("df_test_pos: ", df_test_pos.head(3))
    print("df_test_neg: ", df_test_neg.shape[0], df_test_neg.shape[1])
    print("df_test_neg: ", df_test_neg.head(3))
    print("df_test_neg_baiduqa: ", df_test_neg_baiduqa.shape[0], df_test_neg_baiduqa.shape[1])
    print("df_test_neg_baiduqa: ", df_test_neg_baiduqa.head(3))

    # 组装正负样本为 train 和 test
    df_train = pd.concat([df_train_pos, df_train_neg, df_train_neg_baiduqa], axis=0)
    df_test = pd.concat([df_test_pos, df_test_neg, df_test_neg_baiduqa], axis=0)
    print("df_train: ", df_train.shape[0], df_train.shape[1])
    print("df_train: ", df_train.head(3))
    print("df_train: ", df_train.tail(3))
    print("df_test: ", df_test.shape[0], df_test.shape[1])
    print("df_test: ", df_test.head(3))
    print("df_test: ", df_test.tail(3))
    #
    # 验证下正负样本比
    print("==================================================")
    print("pos-neg in train:", len(df_train[df_train.label==1])/len(df_train[df_train.label==0]))
    print("pos-neg in test:", len(df_test[df_test.label==1])/len(df_test[df_test.label==0]))
    print("==================================================")
    # 再次内部shuffle下
    df_train_shuffle = df_train.sample(frac=1.0, random_state=20230727)
    df_test_shuffle = df_test.sample(frac=1.0, random_state=20230727)
    print("df_train_shuffle: ", df_train_shuffle.shape[0], df_train_shuffle.shape[1])
    print("df_train_shuffle: ", df_train_shuffle.head(3))
    print("df_train_shuffle: ", df_train_shuffle.tail(3))
    print("df_test_shuffle: ", df_test_shuffle.shape[0], df_test_shuffle.shape[1])
    print("df_test_shuffle: ", df_test_shuffle.head(3))
    print("df_test_shuffle: ", df_test_shuffle.tail(3))

    # 写入文件
    df_train_shuffle.to_csv(train_output)  
    df_test_shuffle.to_csv(test_output)
