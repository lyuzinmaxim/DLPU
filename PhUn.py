import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True #работает медленнее, но зато воспроизводимость!


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False,padding_mode='replicate')	

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                     stride=stride, padding=0, bias=False)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = conv1x1(in_channels,out_channels,stride)
        self.conv2 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.LeakyReLU(inplace=False)  
        
    def forward(self, x):
        
        branch = self.conv1(x)
        
        out = self.conv2(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out += branch
        
        return out

def residualblock4creator(in_channels, out_channels):
    return nn.Sequential(
       ResidualBlock(in_channels,out_channels),
       ResidualBlock(in_channels,out_channels),
       ResidualBlock(in_channels,out_channels),
       ResidualBlock(in_channels,out_channels)
    )
		
def up_creator(in_channels, out_channels):
    return nn.Sequential(
       conv3x3(in_channels,in_channels*2),
       nn.LeakyReLU(inplace=False),
       nn.ConvTranspose2d(
                    in_channels=in_channels*2,
                    out_channels=out_channels,
                    kernel_size=2, 
                    stride=2)
    )

class PhUn(torch.nn.Module):
  
  def __init__(self):
    super(PhUn,self).__init__()
    self.conv1 = conv3x3(1,4)
    
    self.block1 = residualblock4creator(4,4)
    self.conv2 = conv3x3(4,8)
    self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.block2 = residualblock4creator(8,8)
    self.conv3 = conv3x3(8,16)
    self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.block3 = residualblock4creator(16,16)
    self.conv4 = conv3x3(16,32)
    self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.block4 = residualblock4creator(32,32)


    self.block_up1 = up_creator(32,16)
    self.block_up2 = up_creator(16,8)
    self.block_up3 = up_creator(8,4)
    
    self.conv5 = conv3x3(4,1)
    self.conv6 = conv1x1(4,1)
    
    #self.conv_out = conv1x1(2,1)

  def forward(self,image):
    
    x = self.conv1(image)

    x = self.block1(x)
    
    branch = self.block1(x) 
    branch = self.conv6(branch)#****
    

    x = self.conv2(x)
    x = self.max_pool_2x2(x)
    
    x = self.block2(x)
    x = self.conv3(x)
    x = self.max_pool_2x2(x)
    
    x = self.block3(x)
    x = self.conv4(x)
    x = self.max_pool_2x2(x)

    x = self.block4(x)
    
    x = self.block_up1(x)
    x = self.block_up2(x)
    x = self.block_up3(x)

    x = self.conv5(x)
    
    x = x + branch
    #print(x.size())
    return x


if __name__ == "__main__":
  image = torch.rand((1,1,256,256))
  model = DLPU()
  print(model(image).size())
