# DLPU
PyTorch model DLPU for phase unwrapping

This is a PyTorch realisation of deep convolutional Unet-like network, described in arcticle [1]. 

Original network was designed in TensorFlow framework, and this is the PyTorch version of it.

I've added following moments to the structure:

1. Replication padding mode in conv3x3 blocks, because experiments have shown that it's important at the edges of phase maps,
otherwise unwrapping quality will be low

# Dataset
Dataset was generated synthetically according to articles [1,2]

So, dataset data was generated using two methods (in equal proportions):

1. Interpolation of squared matrixes (with uniformly distributed elements) of different sizes (2x2 to 15x15) to 256x256 and multiplying by random value, so the magnitude is between 0 and 22 rad
2. Randomly generated Gaussians on 256x256 field with random quantity of functions, means, STD, and ultiplying by random value, so the magnitude is between 2 and 20 rad

![example1](https://user-images.githubusercontent.com/73649419/115595429-95d36d00-a2df-11eb-8d83-1a629635a66f.png)
![example2](https://user-images.githubusercontent.com/73649419/115595433-97049a00-a2df-11eb-95d0-73c631d73240.png)

# References
1. K. Wang, Y. Li, K. Qian, J. Di, and J. Zhao, “One-step robust deep
learning phase unwrapping,” Opt. Express 27, 15100–15115 (2019).
2. Spoorthi, G. E. et al. “PhaseNet 2.0: Phase Unwrapping of Noisy Data Based on Deep Learning Approach.” IEEE Transactions on Image Processing 29 (2020): 4862-4872.
