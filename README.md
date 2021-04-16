# DLPU
PyTorch model DLPU for phase unwrapping

This is a PyTorch realisation of deep convolutional Unet-like network, described in arcticle "K. Wang, Y. Li, K. Qian, J. Di, and J. Zhao, “One-step robust deep
learning phase unwrapping,” Opt. Express 27, 15100–15115 (2019).". 

Original network was designed in TensorFlow framework, and this is the PyTorch version of it.

I've added following moments to the structure:

1. Replication padding mode in conv3x3 blocks, because experiments have shown that it's important at the edges of phase maps,
otherwise unwrapping quality will be low
