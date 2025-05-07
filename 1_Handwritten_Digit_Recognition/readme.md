### 项目结构 📂
- **train**: 数据集下载与处理，模型训练、评估及结果可视化，保存训练好的模型。
- **digit_recognition**: 加载本地保存的模型，对本地手写体数字图片进行单个数字的预测。
- **digit_recognition_multi**: 在 `digit_recognition` 的基础上扩展，对包含多个数字的图片，使用 `opencv` 分割图片后逐一识别并合并结果（效果可能较弱，可以尝试进行优化）。
- **data**: 进行推理测试的图片


### 进阶
[K-12 手写体（HME100K）数据集](https://ai.100tal.com/dataset)  
利用这里的数据集实现 图片-->对应的markdown公式