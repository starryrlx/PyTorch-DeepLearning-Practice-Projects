## 参考文章：
[对抗攻击（Adversarial Attack）](https://blog.csdn.net/ji_meng/article/details/123931315?spm=1001.2014.3001.5501)（基本介绍）   
[【对抗攻击代码实战】对抗样本的生成——FGSM](https://blog.csdn.net/ji_meng/article/details/124366646)（分别基于优化和梯度的白盒攻击示例）  
 
## 目标：
对于一个输入样本 $x$，模型的预测为 $f(x)$，攻击者希望通过添加扰动 $\delta$ 得到对抗样本 $x' = x + \delta$，使得 $f(x') \neq f(x)$ 或达到特定的错误分类。

## 白盒攻击（white-box attack） 
假设攻击者知道模型结构、权重等，针对性通过梯度生成对抗样本  
为了更好地展示攻击效果，这里并没有使用数据增强的操作，把项目二的训练代码拷贝了过来，删除数据增强的操作，重新训练了一个猫狗分类模型用于攻击。
需要先运行`train.ipynb`文件，得到训练好的模型，本项目基于此模型进行攻击。

### 基于单样例的 FGSM 攻击 (Fast Gradient Sign Method)

 
### 基于多样例的 PGD 攻击 (Projected Gradient Descent)


### 黑盒攻击（black-box attack）

不知道模型信息。攻击者只是一个标准用户，只知道模型的输出（标签或置信度分数）