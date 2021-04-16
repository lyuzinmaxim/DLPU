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

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        
        super(ResidualBlock, self).__init__()
        
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        #self.downsample = downsample
        
        
    def forward(self, x):
        
        out = self.conv1(x)
        residual = out
        out = self.bn1(out)
        out = self.relu(out)
        out += residual
        out = self.conv2(out)
        out = self.bn2(out)
        #if self.downsample:
        #    residual = self.downsample(x)
        out = self.relu(out)
        #out = self.downsampling(out)
        return out
		
class ResidualBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        
        super(ResidualBlockUp, self).__init__()
        
        self.conv1 = conv3x3(in_channels, 2*out_channels, stride)
        self.bn1 = nn.BatchNorm2d(2*out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(2*out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        #self.downsample = downsample
        
        
    def forward(self, x):
        
        out = self.conv1(x)
        residual = out
        out = self.bn1(out)
        out = self.relu(out)
        out += residual
        out = self.conv2(out)
        out = self.bn2(out)
        #if self.downsample:
        #    residual = self.downsample(x)
        out = self.relu(out)
        #out = self.downsampling(out)
        return out

class DLPU(torch.nn.Module):
  
  def __init__(self):
    super(DLPU,self).__init__()

    self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.block1 = ResidualBlock(1,8)
    self.block2 = ResidualBlock(8,16)
    self.block3 = ResidualBlock(16,32)
    self.block4 = ResidualBlock(32,64)
    self.block5 = ResidualBlock(64,128)
    self.block6 = ResidualBlock(128,256)

    self.block_up1 = ResidualBlockUp(256,64)
    self.block_up2 = ResidualBlockUp(128,32)
    self.block_up3 = ResidualBlockUp(64,16)
    self.block_up4 = ResidualBlockUp(32,8)
    self.block_up5 = ResidualBlockUp(16,1)

    self.up_trans_1 = nn.ConvTranspose2d(
        in_channels=256,
        out_channels=128,
        kernel_size=2, 
        stride=2)
    
    self.up_trans_2 = nn.ConvTranspose2d(
        in_channels=64,
        out_channels=64,
        kernel_size=2, 
        stride=2)
    
    self.up_trans_3 = nn.ConvTranspose2d(
        in_channels=32,
        out_channels=32,
        kernel_size=2, 
        stride=2)
    
    self.up_trans_4 = nn.ConvTranspose2d(
        in_channels=16,
        out_channels=16,
        kernel_size=2, 
        stride=2)
    
    self.up_trans_5 = nn.ConvTranspose2d(
        in_channels=8,
        out_channels=8,
        kernel_size=2, 
        stride=2)

    self.out = nn.Conv2d(
        in_channels=64,
        out_channels=1,
        kernel_size=1
    )

  def forward(self,image):

    #encoder
    x1 = self.block1(image)
    x2 = self.max_pool_2x2(x1)
    
    x3 = self.block2(x2)
    x4 = self.max_pool_2x2(x3)
    
    x5 = self.block3(x4)
    x6 = self.max_pool_2x2(x5)

    x7 = self.block4(x6)
    x8 = self.max_pool_2x2(x7)

    x9 = self.block5(x8)
    x10 = self.max_pool_2x2(x9)

    #нижняя часть
    x11 = self.block6(x10)

    #decoder
    x = self.up_trans_1(x11)
    x = torch.cat([x,x9],1)
    x = self.block_up1(x)

    x = self.up_trans_2(x)
    x = torch.cat([x,x7],1)
    x = self.block_up2(x)

    x = self.up_trans_3(x)
    x = torch.cat([x,x5],1)
    x = self.block_up3(x)

    x = self.up_trans_4(x)
    x = torch.cat([x,x3],1)
    x = self.block_up4(x)

    x = self.up_trans_5(x)
    x = torch.cat([x,x1],1)
    x = self.block_up5(x)
    return x

    #print(x.size(),'мой вывод после "линии"')

if __name__ == "__main__":
  image = torch.rand((1,1,256,256))
  model = DLPU()
  print(model(image).size())
  

model_DLPU = DLPU()

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

def create_dataset_element(base_size,end_size,magnitude_min,magnituge_max):
  array = np.random.rand(base_size,base_size)
  coef = np.random.permutation(np.arange(magnitude_min,magnituge_max,0.1))[0]
  element = cv2.resize(array, dsize=(end_size,end_size), interpolation=cv2.INTER_CUBIC)
  element = element*coef
  if np.min(element)>=0:
      min_value = np.min(element)
      element = element - min_value
  else:
      min_value = np.min(element)
      element = element + abs(min_value)
  return element
  
#пока предел оперативки (12 ГБ) - 20к  
n = 5000
dataset = np.empty([n,256, 256])

for i in range (n):
  size = np.random.permutation(np.arange(2,15,1))[0]
  dataset[i] = create_dataset_element(size,256,4,20)
  #print(i,'Iteration: created element with base size',size)

#dataset.shape
for i in range(len(dataset[:,0,0])):
  if np.amin(dataset[i,:,:])!=0:
    print(i,'error here')

dataset_torch = torch.from_numpy(dataset)
dataset_unsqueezed = dataset_torch.unsqueeze(1).float()
dataset_unsqueezed.shape

def wraptopi(input):
  pi = 3.1415926535897932384626433;
  output = input - 2*pi*np.floor( (input+pi)/(2*pi) );
  return (output)

X = wraptopi(dataset_unsqueezed);

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(
    X[:,:,:,:], 
    dataset_unsqueezed[:,:,:,:], 
    test_size=0.3, 
    shuffle=True)

print(X_train.shape,'Размерность тренировочных картинок "wraped phase"')
print(X_test.shape,'Размерность тестовых картинок "wraped phase"')
print(Y_train.shape,'Размерность тренировочных картинок ground truth')
print(Y_test.shape,'Размерность тестовых картинок ground truth')


def au_and_bem_torch(nn_output,ground_truth,calc_bem: bool):
    '''
    difference from "au_and_bem' is converting to np.ndarray and abs()

    calculates Binary Error Map (BEM) and Accuracy of Unwrapping (AU)
    for batch [batch_images,0,width,heidth] and returns mean AU of a batch
    with list of AU for every image and may be with BEM (optionally)

    function returns:
    au_mean - float, mean AU for batch
    au_list - list, info about AU for every image in batch
    bem - 3d boolean tensor, shows BEM in format [images_in_batch,width,height]

    args:
    nn_output - ndarray or torch.tensor - tensor that goes forward the net
    ground_truth - ndarray or tensor - ground truth image (original phase)
    calc_bem - boolean, if needed, will calculate BEM
    
    
    with input as np.ndarray runs 10 times faster
    '''
    nn_output = nn_output.numpy()
    ground_truth = ground_truth.numpy()
    
    au_list = []
    bem = np.empty([
        len(nn_output[:,0,0,0]),
        len(nn_output[0,0,:,0]),
        len(nn_output[0,0,0,:])
    ])
    
    for k in range(len(nn_output[:,0,0,0])):
        min_height = 0
        cnt = 0
        for i in range(len(nn_output[0,0,:,0])):
            for j in range(len(nn_output[0,0,0,:])):
                x = abs(nn_output[k,0,i,j]-ground_truth[k,0,i,j])
                
                if calc_bem:
                    
                    if x <= (ground_truth[k,0,i,j] - min_height)*0.05:
                        bem[k,i,j] = 1
                        cnt +=1
                    else:
                        bem[k,i,j] = 0
                
                else:
                    if x <= (ground_truth[k,0,i,j] - min_height)*0.05:
                        cnt +=1
                                          
        au = cnt/(len(nn_output[0,0,:,0])*len(nn_output[0,0,0,:]))
        #print(k,'au:',au)
        au_list.append(au)
    
    au_mean = sum(au_list)/len(au_list)
    
    if calc_bem:
        return au_mean,au_list,bem
    else:
        return au_mean,au_list
        
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def model_train(
    model,
    name,
    batch_size,
    total_epochs,
    learning_rate,
    loss_freq,
    metric_freq,
    lr_freq,
    save_freq):
  
    '''
    That function makes train process easier, only optimizer hyperparameters 
    shoud be defined in function manually

    function returns:
    1. trained model
    2. list of metric history for every "metric_freq" epoch
    3. list of losses history for every "loss_freq" epoch

    args:
    model - torch.nn.Module object - defined model 
    name - string, model checkpoints will be saved with this name
    batch size - integer, defines number of images in one batch
    total epoch - integer, defines number of epochs for learning
    learning rate - float, learning rate of an optimizer
    loss_freq - integer, loss function will be computed every "loss_freq" epochs
    metric_freq - integer, metric (AU) -||-
    lr_freq - intereger, learning rate will be decreased -||-
    save_freq - integer, model checkpoints for train and validation will 
                be saved  -||-
    
    *time computing supports only GPU calculations
    ''' 
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # 0 обозначает первую по порядку видеокарту

    if device.type == 'cuda':
      start = torch.cuda.Event(enable_timing=True)
      end = torch.cuda.Event(enable_timing=True)

    model = model.to(device)
    print('[INFO] Model will be learned on {}'.format(device)) 
    
    metric_history = []
    test_loss_history = []
    
    loss = torch.nn.L1Loss(size_average=None, reduce=None, reduction='mean')
    #loss = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    
    if device.type == 'cuda':
        start.record()

    for epoch in np.arange(0,total_epochs,1):
      
      print('>> Epoch: {}/{} Learning rate: {}'.format(epoch,total_epochs,learning_rate))

      order = np.random.permutation(len(X_train))

      for start_index in range(0, len(X_train), batch_size):
          
          optimizer.zero_grad()
          model.train()   
          batch_indexes = order[start_index:start_index+batch_size]

          X_batch = X_train[batch_indexes].to(device)
          Y_batch = Y_train[batch_indexes].to(device)
          
          preds = model.forward(X_batch) 
          
          loss_value = loss(preds, Y_batch)
          loss_value.backward()
          
          optimizer.step()
          ##### memory optimization start #####
          #GPUtil.showUtilization()

          del X_batch,Y_batch
          torch.cuda.empty_cache()
          
          #GPUtil.showUtilization()
          ##### memory optimization end #####

      if epoch % loss_freq == 0:
          test_per_batch = []
          print('[INFO] beginning to calculate loss')
          model.eval()
          order_test = np.random.permutation(len(X_test))

          for start_index_test in range(0, len(X_test), batch_size):
              test_per_batch = []
    
              batch_indexes_test = order_test[start_index_test:start_index_test+batch_size]
    
              with torch.no_grad(): 
                X_batch_test = X_test[batch_indexes_test].to(device)
                Y_batch_test = Y_train[batch_indexes_test].to(device)
      
                test_preds = model.forward(X_batch_test)
                metric_loss = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
                
                test_loss = metric_loss(test_preds, Y_batch_test)
                test_per_batch.append(test_loss.data.cpu())

                ##### memory optimization start #####
                del X_batch_test,Y_batch_test
                torch.cuda.empty_cache()
                ##### memory optimization end #####

          test_loss_epoch = sum(test_per_batch) / len(test_per_batch)
          test_loss_history.append(test_loss_epoch.tolist())
          
          print('[LOSS] mean value of MSE {:.4f} at epoch number {}'.format(test_loss_epoch,epoch))    
      
      if epoch % metric_freq ==0:
          model.eval()

          order_metric = np.random.permutation(len(X_test))

          for start_index_metric in range(0, len(X_test), batch_size):
              metric_per_batch = []

              batch_indexes_metric = order_metric[start_index_metric:start_index_metric+batch_size]
              
              with torch.no_grad():
                  X_batch_metric = X_test[batch_indexes_metric].to(device)

                  Y_batch_metric = Y_test[batch_indexes_metric]

                  metric_preds = model.forward(X_batch_metric)
                  
                  #mean_au,_ = au_and_bem_torch(Y_batch_metric,metric_preds.detach().to('cpu'),calc_bem=False)
                  mean_au_batch,_ = au_and_bem_torch(metric_preds.detach().to('cpu'),Y_batch_metric,calc_bem=False)

                  metric_per_batch.append(mean_au_batch)
                  #metric_per_batch.append(mean_au_batch.data.cpu())

                  ##### memory optimization start #####
                  #GPUtil.showUtilization()
                  del X_batch_metric,Y_batch_metric,metric_preds
                  torch.cuda.empty_cache()
                  #GPUtil.showUtilization()    
                  ##### memory optimization end #####
                  
          test_metric_epoch = sum(metric_per_batch) / len(metric_per_batch)
          metric_history.append(test_metric_epoch)
          print('[METRIC] Accuracy of unwrapping on test images is {:.4f} %,'.format(test_metric_epoch*100))

      if (epoch + 1) % save_freq == 0:
          torch.save({
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'loss': loss
              }, '/root/model/{}_checkpoint_{}'.format(name,(epoch+1)))
          print('[SAVE] /root/model/{}_checkpoint_{} was saved'.format(name,(epoch+1)),)
          
      if (epoch + 1) % lr_freq == 0:
          learning_rate /= 2
          update_lr(optimizer, learning_rate)
          print('[lr]New learning rate: {}'.format(learning_rate))

    print('[END]Learning is done')
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
                #,'lr': learning_rate
                }, '/root/model/{}_checkpoint_end'.format(name))
    print('[END]/root/model/{}_checkpoint_end was saved'.format(name))
    
    if device.type == 'cuda':
      end.record()
      torch.cuda.synchronize()
      print('Learning time is {:.1f} min'.format(start.elapsed_time(end)/(1000*60)))
    
    import csv
    with open('/root/model/metric_{}.csv'.format(name), 'w', newline='') as myfile:
      wr = csv.writer(myfile, quoting=csv.QUOTE_NONE)
      wr.writerow(metric_history)
    
    with open('/root/model/loss_{}.csv'.format(name), 'w', newline='') as myfile:
      wr = csv.writer(myfile, quoting=csv.QUOTE_NONE)
      wr.writerow(test_loss_history)


    return model,metric_history,test_loss_history


#model_VUR_Net = model_VUR_Net.half()
import csv
trained_model1,list_metric1,list_loss1 = model_train(
                                            model=model_DLPU,
                                            name='DLPU-GoogleCloud',
                                            batch_size=16,
                                            total_epochs=500,
                                            learning_rate=0.0002,
                                            loss_freq=1,
                                            metric_freq=1,
                                            lr_freq=10,
                                            save_freq=10)
