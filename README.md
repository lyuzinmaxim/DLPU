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
2. Randomly generated Gaussians on 256x256 field with random quantity of functions, means, STD, and multiplying by random value, so the magnitude is between 2 and 20 rad

![Example1](https://user-images.githubusercontent.com/73649419/116145971-9fe1db00-a6e6-11eb-9ff3-7afc4982f8a3.png)
![Example2](https://user-images.githubusercontent.com/73649419/116145975-a1130800-a6e6-11eb-8b57-5cbf2e168ac9.png)



# Metrics
I've implemented BEM (Binary Error Map), described in [3] with threshold 5%, according to formula

![render](https://user-images.githubusercontent.com/73649419/116073854-a5650400-a699-11eb-9dbd-30510f355bb6.png)


# References
1. K. Wang, Y. Li, K. Qian, J. Di, and J. Zhao, “One-step robust deep
learning phase unwrapping,” Opt. Express 27, 15100–15115 (2019).
2. Spoorthi, G. E. et al. “PhaseNet 2.0: Phase Unwrapping of Noisy Data Based on Deep Learning Approach.” IEEE Transactions on Image Processing 29 (2020): 4862-4872.
3. Qin, Y., Wan, S., Wan, Y., Weng, J., Liu, W., & Gong, Q. (2020). Direct and accurate phase unwrapping with deep neural network. Applied optics, 59 24, 7258-7267 .
