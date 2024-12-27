参考文章：
[文本分类原理与实践](https://aibydoing.com/notebooks/chapter09-04-lab-principles-and-practices-of-text-classification) 精读

[预训练模型分类](https://juejin.cn/post/7327721919426494474#heading-3)

[NLP之文本分类：「Tf-Idf、Word2Vec和BERT」三种模型比较](https://www.leiphone.com/category/yanxishe/tbzazc3cjams815p.html)


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


### 训练迭代过程

- lr=1e-4, epochs=10 : Train_acc:69.7%, Train_loss:0.590, Test_acc:67.9%，Test_loss:0.599  训练不充分，没有收敛
- lr=1e-4, epochs=20 : Train_acc:72.2%, Train_loss:0.529, Test_acc:66.4%，Test_loss:0.611  有点过拟合了，准确率也很低
- lr=1e-3, epochs=10 : Train_acc:73.9%, Train_loss:0.501, Test_acc:66.7%，Test_loss:0.628  准确率略有提升
- lr=1e-3, epochs=10使用StepLR动态调整学习率 step_size=2, gamma=0.5 : Train_acc:75.0%, Train_loss:0.484, Test_acc:68.1%，Test_loss:0.611 训练集正确率很高，测试集准确率很有差距，很明显的过拟合
前面四种方法只能提升Train_acc，但是Test_acc没有提升，可能是因为模型参数量太少
句子长度：max_length 50 -> 64; Word2Vec window 5 -> 8; self.lstm num_layers=2, dropout=0.3, 增加一个全连接层  (模型可训练参数数量: 230000 -> 660000)
Train_acc:74.4%, Train_loss:0.495, Test_acc:68.1%，Test_loss:0.610

max_length=64 -> 100, vector_size=128 -> 256, dropout=0.5 (模型可训练参数数量: 660000 -> 2760000)  Train_acc:76.0%, Train_loss:0.465, Test_acc:68.1%，Test_loss:0.640  为什么Test_acc就是提不上去？

上点强度：训练嵌入层 self.embedding.weight.requires_grad = True (模型可训练参数数量: 2760000 -> 56,480000) Train_acc一路飙到了94.0%，但是Test_acc不增反降：Train_acc:94.0%, Train_loss:0.124, Test_acc:63.2%，Test_loss:2.856

我们模型的词表太大了，导致整个模型可学习参数都在embedding层，尝试减少词表大小 min_count=1 -> min_count=4 : (词表大小： 模型可训练参数数量: 56,480000 -> )

到这里就尽力啦，留待日后再探索优化


本来随手加个一个正则化，结果跑出来测试集和验证集都是50%的正确率，直接把模型搞废了，给我吓了一大跳 optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=)试了0.01,1e-5,1e-6, 应该是因为惩罚权重导致模型参数接近于0失去了学习能力？