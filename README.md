# DLPU
PyTorch model DLPU for phase unwrapping

This is a PyTorch realisation of deep convolutional Unet-like network, described in arcticle [1]. 

Original network was designed in TensorFlow framework, and this is the PyTorch version of it.

# Changes
I've added following moments to the structure:

1. Replication padding mode in conv3x3 blocks, because experiments have shown that it's important at the edges of phase maps,
otherwise unwrapping quality will be low
2. In article there are some unclear moments: neural net structure contains of "five repeated uses of two 3×3 convolution operations (each followed by a BN and a ReLU), a residual block between the two convolution operations,..."
So, according to the article it should be CONV3x3->BN->ReLU -> Residual Block(???) -> CONV3x3->BN->ReLU and it's not clear. In contracting path (down) it's possible to make "good" residual connection, as shown below
![ResBlock_contracting](https://user-images.githubusercontent.com/73649419/116404556-6d93c300-a837-11eb-92ba-64a560383338.jpg)

But autors write, that in expansive path there is similar structure CONV3x3->BN->ReLU -> Residual Block(???) -> CONV3x3->BN->ReLU and it's impossible to use residual connection below (figure from article)
![1](https://user-images.githubusercontent.com/73649419/116405461-599c9100-a838-11eb-9405-8d951600ab35.jpg)
because first CONV3x3 reduces channels by two, and second CONV3x3 reduces again channels by two, and that makes no sence (and possibility, because number of channels don't match) to use residual connection here

![ResBlock_expansive](https://user-images.githubusercontent.com/73649419/116405910-d7f93300-a838-11eb-9e8d-352c3719344e.jpg)


# Dataset
Dataset was generated synthetically according to articles [1,2]

So, dataset data was generated using two methods (in equal proportions):

1. Interpolation of squared matrixes (with uniformly distributed elements) of different sizes (2x2 to 15x15) to 256x256 and multiplying by random value, so the magnitude is between 0 and 22 rad
2. Randomly generated Gaussians on 256x256 field with random quantity of functions, means, STD, and multiplying by random value, so the magnitude is between 2 and 20 rad

![Example1](https://user-images.githubusercontent.com/73649419/116145971-9fe1db00-a6e6-11eb-9ff3-7afc4982f8a3.png)
![Example2](https://user-images.githubusercontent.com/73649419/116145975-a1130800-a6e6-11eb-8b57-5cbf2e168ac9.png)

# Model
Model can be shown as following:
In original paper there is unclear moment: "residual block (see Ref.
20 for details) between the two convolution operations"
![DLPU structure](https://user-images.githubusercontent.com/73649419/116402297-e8a7aa00-a834-11eb-901d-d400f1b46a65.png)



# Metrics
I've implemented BEM (Binary Error Map), described in [3] with threshold 5%, according to formula

![render](https://user-images.githubusercontent.com/73649419/116073854-a5650400-a699-11eb-9dbd-30510f355bb6.png)


# References
1. K. Wang, Y. Li, K. Qian, J. Di, and J. Zhao, “One-step robust deep
learning phase unwrapping,” Opt. Express 27, 15100–15115 (2019).
2. Spoorthi, G. E. et al. “PhaseNet 2.0: Phase Unwrapping of Noisy Data Based on Deep Learning Approach.” IEEE Transactions on Image Processing 29 (2020): 4862-4872.
3. Qin, Y., Wan, S., Wan, Y., Weng, J., Liu, W., & Gong, Q. (2020). Direct and accurate phase unwrapping with deep neural network. Applied optics, 59 24, 7258-7267 .
