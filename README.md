# DLPU
PyTorch model DLPU for phase unwrapping

This is a PyTorch realisation of deep convolutional Unet-like network, described in arcticle "K. Wang, Y. Li, K. Qian, J. Di, and J. Zhao, “One-step robust deep
learning phase unwrapping,” Opt. Express 27, 15100–15115 (2019).". 

Original network was designed in TensorFlow framework, and this is the PyTorch version of it.

I've added following moments to the structure:

1. Replication padding mode in conv3x3 blocks, because experiments have shown that it's important at the edges of phase maps,
otherwise unwrapping quality will be low

# Dataset
Dataset was generated synthetically according to article "K. Wang, Y. Li, K. Qian, J. Di, and J. Zhao, “One-step robust deep
learning phase unwrapping,” Opt. Express 27, 15100–15115 (2019)." and "Spoorthi, G. E. et al. “PhaseNet 2.0: Phase Unwrapping of Noisy Data Based on Deep Learning Approach.” IEEE Transactions on Image Processing 29 (2020): 4862-4872."

So, dataset data was generated using two methods (in equal proportions):

1. Interpolation of squared matrixes (with uniformly distributed elements) of different sizes (2x2 to 15x15) to 256x256 and multiplying by random value, so the magnitude is between 0 and 22 rad
2. Randomly generated Gaussians on 256x256 field with random quantity of functions, means, STD, and ultiplying by random value, so the magnitude is between 2 and 20 rad

![1](https://user-images.githubusercontent.com/73649419/115594966-07f78200-a2df-11eb-8537-3e0189b45a87.png)
![1wraped](https://user-images.githubusercontent.com/73649419/115594974-0a59dc00-a2df-11eb-8ff1-9842f7af5302.png)
![2](https://user-images.githubusercontent.com/73649419/115594982-0c239f80-a2df-11eb-8c61-98ffd923ee0f.png)
![2wraped](https://user-images.githubusercontent.com/73649419/115594989-0ded6300-a2df-11eb-81a4-71aa60ae4faf.png)
