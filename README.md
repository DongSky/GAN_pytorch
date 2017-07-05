# GAN_pytorch
GAN/WGAN written by PyTorch, using CIFAR10 dataset. Just a practice. 

### How to run
first, install Pytorch. Compiling from source or using Binary, either is OK. 
for detail please see https://pytorch.org/

then, execute python gan.py or python wgan.py

### Others
In these programs, I found that 30 epoches in GAN and 130 epoches in WGAN might be a good choice, and the quality of generator will decline with the number of epoch increasing. in WGAN maybe because of the cliping operation because the paramaters changed to some value near 0.01 or -0.01. 
