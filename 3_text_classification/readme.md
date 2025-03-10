参考文章：
[文本分类原理与实践](https://aibydoing.com/notebooks/chapter09-04-lab-principles-and-practices-of-text-classification) 

[预训练模型分类](https://juejin.cn/post/7327721919426494474#heading-3)

[NLP之文本分类：「Tf-Idf、Word2Vec和BERT」三种模型比较](https://www.leiphone.com/category/yanxishe/tbzazc3cjams815p.html)

本项目所使用的数据集是作者从豆瓣爬取的，一共有50w条正负样本均衡的数据。

提升模型性能的几个关键调整方向：

1. 数据层面：
- 增加 max_length (如设为 100)，避免截断重要信息
- 调整 Word2Vec 参数：增加 vector_size（如 200）和 window size（如 8）

2. 模型层面：
```python
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, embedding_matrix, hidden_dim=256):  # 增加 hidden_dim
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix))
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)  # 增加层数
        self.attention = nn.Linear(hidden_dim * 2, 1)  # 添加注意力机制
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)  # 添加全连接层
        self.fc2 = nn.Linear(hidden_dim, 2)
        self.dropout = nn.Dropout(0.5)  # 增加 dropout
```

3. 训练参数：
- 降低学习率：`optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)`
- 使用学习率调度器：`scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)`
- 增加训练轮次到 30-50 epochs
- 实现早停机制，避免过拟合


训练结果不是很好，有可能是数据的问题，数据本身质量不高或者划分正负例方式有问题