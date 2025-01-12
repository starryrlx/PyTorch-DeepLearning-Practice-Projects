参考文章：[GAN 原理 & pytorch代码实例 - 生成MINIST手写数字](https://blog.csdn.net/Lizhi_Tech/article/details/132108893)


更多：
图像修复：修复损坏或缺失的图像
超分辨率：将低分辨率图像转换为高分辨率图像


#### SNGAN
为了解决GAN训练不稳定的问题，采用SNGAN（Spectral Normalization GAN）。
SNGAN的主要思想是通过对生成器和判别器的权重进行谱归一化，使得权重的分布更加平滑，从而提高模型的稳定性。

通过将判别器改为 SNGAN 并使用 Hinge Loss，可以有效稳定 GAN 的训练过程，避免生成器损失越来越大的问题。
