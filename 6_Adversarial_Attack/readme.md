### 参考文章：
[对抗攻击（Adversarial Attack）](https://blog.csdn.net/ji_meng/article/details/123931315?spm=1001.2014.3001.5501)（基本介绍）   
[【对抗攻击代码实战】对抗样本的生成——FGSM](https://blog.csdn.net/ji_meng/article/details/124366646)（分别基于优化和梯度的白盒攻击示例）  
 
### 目标：
对于一个输入样本 $x$，模型的预测为 $f(x)$，攻击者希望通过添加扰动 $\delta$ 得到对抗样本 $x' = x + \delta$，使得 $f(x') \neq f(x)$ 或达到特定的错误分类。

### 白盒攻击（white-box attack） 

假设攻击者知道模型结构、权重等，针对性通过梯度生成对抗样本  
本项目针对[猫狗图像分类 (Cat & Dog Image Classification)](../2_Cat_Dog_Image_Classification/)中训练的模型进行攻击  
狗的预测结果很容易改变，但是作者无论怎么改参数猫图片预测都是正确的。



### 黑盒攻击（black-box attack）

不知道模型信息。攻击者只是一个标准用户，只知道模型的输出（标签或置信度分数）