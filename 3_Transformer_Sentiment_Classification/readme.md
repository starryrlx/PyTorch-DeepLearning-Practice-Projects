参考文章：
[文本分类原理与实践](https://aibydoing.com/notebooks/chapter09-04-lab-principles-and-practices-of-text-classification) 

[预训练模型分类](https://juejin.cn/post/7327721919426494474#heading-3)

[NLP之文本分类：「Tf-Idf、Word2Vec和BERT」三种模型比较](https://www.leiphone.com/category/yanxishe/tbzazc3cjams815p.html)

本项目所使用的数据集是作者从豆瓣爬取的，一共有50w条正负样本均衡的数据。  

### 数据集
- 来源：豆瓣电影评论，包含用户评分（1-5 星）和评论文本。
- 规模：508,110 条记录，包含正面（label=1）和负面（label=0）情感标签。
- 样本示例：
  - 正面：“虽然不长 但是真的不错” (4 星)
  - 负面：“本該是三個騙子互相拆台的一出好戲，變成了兩個傻子在一個騙子的瘋狂擺布下…” (2 星)


一开始参考上面的文章使用了 word2vec 和 Bert 两种方式，效果都不好。最后仅保留了使用 Transformer  



需要手动下载 [bert-base-chinese](https://huggingface.co/google-bert/bert-base-chinese/tree/main) 的分词器文件，并将其放置在指定目录`（../models/3_Chinese_Movie_review_Text_Classification/bert-base-chinese）`。  
bert-base-chinese 分词器需要以下文件：
- vocab.txt
- config.json
- tokenizer_config.json