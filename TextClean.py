import re, jieba


class TextProcessor():
     '''
      - NLP入门-- 文本预处理Pre-processing
      - 用python进行精细中文分句(基于正则表达式)，HarvestText:文本挖掘和预处理工具 - 汀、人工智能 - 博客园

     '''
     def __init__(self, path_stopwords=None, path_user_cutwords=None) -> None:
          self.path_stopwords = path_stopwords
          self.path_user_cutwords = path_user_cutwords
     
     # 1）中文分句
     def cut_sentences_v1(self, text):
          '''
          # ref link: 用python进行精细中文分句(基于正则表达式)，HarvestText:文本挖掘和预处理工具 - 汀、人工智能 - 博客园
          # ref link: https://github.com/blmoistawinde/HarvestText/blob/master/harvesttext/harvesttext.py#LL694C9-L694C22
          '''
          # 中英文分句
          text = re.sub('([，。！？\?])([^”’])', r"\1\n\2", text)  # 单字符断句符：(中文 逗号、句号、感叹号、问号；英文问号), (不匹配中文单双引号”’)
          text = re.sub('(\.{6})([^”’])', r"\1\n\2", text)  # 英文省略号：(英文省略号), (不匹配中文单双引号”’)
          text = re.sub('(\…{2})([^”’])', r"\1\n\2", text)  # 中文省略号：(中文省略号), (不匹配中文单双引号”’)
          # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
          text = re.sub('([，。！？\?][”’])([^，。！？\?])', r'\1\n\2', text) # (中文逗号、句号、感叹号、问号；英文问号), (中文单双引号”’)
          text = text.rstrip()  # 段尾如果有多余的\n就去掉它
          # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
          
          return text.split("\n")
     
     # 2）文本预处理清洗
     # 2.1) 去除数据中的非文本部分
     def remove_nonch_v1(self, sentence):
          # 过滤不了\\ \ 中文（）还有————
          r1 = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'#用户也可以在此进行自定义过滤字符 
          # 者中规则也过滤不完全
          r2 = "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+"
          # \\\可以过滤掉反向单杠和双杠，/可以过滤掉正向单杠和双杠，第一个中括号里放的是英文符号，第二个中括号里放的是中文符号，第二个中括号前不能少|，否则过滤不完全
          r3 =  "[.!//_,$&%^*()<>+\"'?@#-|:~{}]+|[——！\\\\，。=？、：“”‘’《》【】￥……（）]+" 
          # 去掉括号和括号内的所有内容
          r4 =  "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"

          # sentence = "hello! wo?rd!."
          cleanr = re.compile('<.*?>')
          sentence = re.sub(cleanr, ' ', sentence)        #去除html标签
          sentence = re.sub(r1, '',sentence)
          sentence = re.sub(r2, '',sentence)
          sentence = re.sub(r3, '',sentence)
          sentence = re.sub(r4, '',sentence)
          
          return sentence

     def remove_nonch_v2(self, sentence):
          # 过滤不了\\ \ 中文（）还有————
          r1 = u'[a-zA-Z’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'#用户也可以在此进行自定义过滤字符 
          # 者中规则也过滤不完全
          r2 = "[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）]+"
          # \\\可以过滤掉反向单杠和双杠，/可以过滤掉正向单杠和双杠，第一个中括号里放的是英文符号，第二个中括号里放的是中文符号，第二个中括号前不能少|，否则过滤不完全
          r3 =  "[.!//_,$&%^*()<>+\"'?@#-|:~{}]+|[——！\\\\，。=？、：“”‘’《》【】￥……（）]+" 
          # 去掉括号和括号内的所有内容
          r4 =  "\\【.*?】+|\\《.*?》+|\\#.*?#+|[.!/_,$&%^*()<>+""'?@|:~{}#]+|[——！\\\，。=？、：“”‘’￥……（）《》【】]"

          # sentence = "hello! wo?rd!."
          cleanr = re.compile('<.*?>')
          sentence = re.sub(cleanr, ' ', sentence)        #去除html标签
          sentence = re.sub(r1, '',sentence)
          sentence = re.sub(r2, '',sentence)
          # sentence = re.sub(r3, '',sentence)
          sentence = re.sub(r4, '',sentence)
          
          return sentence
     
     # 2.2) 分词. 基于jieba分词
     def cut_words_v1(self, sentence):
          # 加载自定义词典
          if self.path_user_cutwords is not None:
               jieba.load_userdict(self.path_user_cutwords) # file_name 为文件类对象或自定义词典的路径
          sentence_seg = jieba.cut(sentence)
          
          return sentence_seg
     
     # 2.3) 去掉停用词. 基于开源的常用的中文停用词表，1208个停用词
     def remove_stopw_v1(self, word_list, path_stopwords):
          # - https://zhuanlan.zhihu.com/p/335347401
          filter_sentence, stopwords = word_list, []
          # 加载字典
          with open(path_stopwords, "r") as fp:
               stopwords = fp.readlines()
          # 过滤
          filter_sentence = [w for w in word_list if w not in stopwords]
          
          return filter_sentence

     # 得到一个段落的全部不重复分词结果, 返回一个word list
     def text_preprocess_v1(self, text):
          ret_list = []
          path_stopwords = self.path_stopwords

          # 中文分句
          sents_list = self.cut_sentences_v1(text)
          # print("sents_list: ", sents_list)
          # 
          for sents in sents_list:
               # 去除数据中的非文本部分
               sents = self.remove_nonch_v1(sents)
               # print("sents: ", sents)
               # 分词
               word_list = self.cut_words_v1(sents)
               # print("word_list: ", word_list)
               # 去掉停用词
               word_list = self.remove_stopw_v1(word_list, path_stopwords)
               # print("word_list: ", word_list)
               # 汇合所有词为一个list
               for word in word_list:
                    ret_list.append(word)
          ret_list = list(set(ret_list)) # 利用set去重
          # print("ret_list: ", ret_list)
          
          return ret_list

     # 输入一个段落，输出清洗后的一个句子构成的list
     def text_preprocess_v2(self, text):
          ret_list = []
          path_stopwords = self.path_stopwords

          # 中文分句
          sents_list = self.cut_sentences_v1(text)
          # print("sents_list: ", sents_list)
          # 
          for sents in sents_list:
               # 去除数据中的非文本部分
               sents = self.remove_nonch_v2(sents)
               # print("sents: ", sents)
               # 分词
               word_list = self.cut_words_v1(sents)
               # print("word_list: ", word_list)
               # 去掉停用词
               word_list = self.remove_stopw_v1(word_list, path_stopwords)
               # print("word_list: ", list(word_list))
               # 汇合所有词为一个sentence
               if len(word_list) > 0:
                    ret_list.append(''.join(word_list))
               # print("=====")
          # print("ret_list: ", ret_list)
          
          return ret_list
     

def test_text_process():
     path_stopwords = './lib_resource/stopwords/baidu_stopwords.txt'
     text_pro = TextProcessor(path_stopwords=path_stopwords)
     # sents = '''漂亮 的姑娘
     #           善良 的姑娘
     #           聪明 的姑娘朴素 的姑娘能干 的姑娘灵巧 的姑娘机敏 的姑娘机灵 的姑娘认真 的姑娘谨慎 的姑娘
     #           尽职 的姑娘
     #           坚强 的姑娘 直率 的姑娘虚心 的姑娘辛勤 的姑娘刻苦 的姑娘勤劳 的姑娘 俭朴 的姑娘。起床歌 小宝宝,起得早;睁开眼,嘻嘻笑; 咿呀呀,学说话;伸伸手,要人抱。
     #           穿衣歌 小胳膊,穿袖子;穿上衣,扣扣子; 小脚丫,穿裤子;穿上袜子穿鞋子。
     #           小镜子 小镜子,圆又圆;看宝宝,露笑脸; 闭上眼,做个梦;变月亮,挂上天。
     #           '''
     # sents = '''
     #      12304主动打来的原因主要包括：
     #      1. 服务提醒：铁路客服人员会通过电话提醒用户重要事项。
     #      2. 售后服务：客服人员会主动联系用户，提供帮助和指导。
     #      3. 特殊情况通知：如列车延误、停运、线路调整等，会告知调整信息。
     #      1993年电影《东成西就》演员表及角色描述：
     #          黄药师（张国荣 饰）——男主角，素球青梅竹马师兄。
     #          三公主（林青霞 饰）——被夺王位，下山寻找段王爷。
     #      '''
     # sents = '''
     #      电影《1980年代的爱情》改编自郑世平同名小说，讲述了1980年代大学生关雨波与老同学成丽雯之间的爱情故事。关暗恋雯多年，但雯却因身份悬殊而无果。四年后，两人重逢，但彼此疏远。关每天黄昏到小店买酒消愁，雯却指责他的酗酒颓废并暗中关照他的身体。两人开始靠近，但雯仅在生活中关心他，鼓励他重新振作，争取考研回省城以便与恋人团聚。最终，雯选择留在小镇，与关一起面对生活的琐碎和爱情的曲折。
     #      '''
     # sents = '''
     #      1《以爱为营》导演：郭虎主演：白鹿、王鹤棣、魏哲鸣等改编自翘摇小说《错撩》，财经记者郑书意VS霸道总裁之间阴差阳错的爱情故事。
     #      2《南风知我意》导演：李昂编剧：七微、 勺子、周萌、王莹菲主演：成毅、张予曦、付辛博、李欣泽、闫笑等根据七微的同名小说改编，讲述了为了寻找天然药物的药物研发员傅云深与前往欠发达地区进行医学调研的外科医生朱旧相...
     #      3《偷偷藏不住》导演：沙维琪主演：赵露思、陈哲远、马伯骞、曾黎、邱心志、管梓净、王洋、张皓伦等青春校园偶像剧！根据竹已的同名小说改编，腹黑青年段嘉许和乖戾少女相聚之间的爱情故事。
     # '''
     # sents = '''
     #      以下是几首Techno代表作：
     #      Chained to a Dead Camel (2013)
     #      Bring (2013)
     #      Randomer
     #      Steady Note (2014)
     # '''
    #  sents = '''
    #       以下是一些不血腥的谍战剧推荐：

    #       1. 《悬崖》 - 张嘉译、周乙主演的谍战剧，结局令人不尽如人意，但是情节紧凑，人物形象鲜明。

    #       2. 《黎明之前》 - 吴秀波、林永健主演的国产谍战剧，老戏骨演技过硬，剧情流畅，最后的结局令人印象深刻。

    #       3. 《密战》 - 张译、黄志忠、潘之琳、薛佳凝主演的年代反特谍战剧，剧情紧凑，人物性格分明，推荐观看。

    #       4. 《渗透》 - 沙溢、陈瑾、于越、曹炳琨、张佳宁、韩童生主演的轻松谍战剧，情节好看，演员颜值和演技在线。

    #       希望以上推荐能满足您的需求！
    #  '''
     sents = '''
          1-100 年的结婚纪念日名称如下：

          1年 - 纸婚
          2年 - 棉婚
          3年 - 皮革婚
          4年 - 丝绸婚
          5年 - 木婚
          6年 - 糖婚
          7年 - 铜婚
          8年 - 铁婚
          9年 - 陶瓷婚
          10年 - 玫瑰婚
          11年 - 钢婚
          12年 - 镍婚
          13年 - 紫罗兰婚
          14年 - 象牙婚
          15年 - 水晶婚
          16年 - 蓝宝石婚
          17年 - 兰婚
          19年 - 珠母婚
          20年 - 瓷婚
          21年 - 猫眼石婚
          22年 - 青铜婚
          23年 - 钛婚
          24年 - 缎婚
          25年 - 银婚
          26年 - 宝石婚
          27年 - 红木婚
          28年 - 丁香婚
          29年 - 蜜桃婚
          30年 - 珍珠婚
          31年 - 菩提树婚
          32年 - 青金石婚
          33年 - 锡婚
          34年 - 琥珀婚
          35年 - 亚麻布婚
          36年 - 绿宝石婚
          37年 - 孔雀婚
          38年 - 焰火婚
          39年 - 太阳婚
          40年 - 红宝石婚
          41年 - 桦木婚
          42年 - 石榴石婚
          43年 - 铅婚
          44年 - 星辰婚
          45年 - 黄铜婚
          46年 - 薰衣草婚
          47年 - 开士米婚
          48年 - 紫水晶婚
          49年 - 雪松婚
          50年 - 金婚
          51年 - 牧婚
          52年 - 黄玉婚
          53年 - 铀婚
          54年 - 宙斯婚
          55年 - 铂金婚
          60年 - 钻石婚
          70年 - 恩婚
          75年 - 珍宝婚
          80年 - 橡树婚
          85年 - 天使之婚
          90年 - 大理石婚
          100年 - 天婚/天堂婚

          希望以上回答能满足您的需求。
          '''
     ret_list = text_pro.text_preprocess_v2(sents)
     print("ret_list: ", ret_list)
     print("ret_text: ", ','.join(ret_list))

if __name__ == "__main__":
     test_text_process()