from DLPU import *
from dataset_generation import *
from train import *
import csv


n = 5000
dataset = np.empty([n,256, 256])
for i in range (n):
  size = np.random.permutation(np.arange(2,15,1))[0]
  dataset[i] = create_dataset_element(size,256,4,20)
 
dataset_torch = torch.from_numpy(dataset)
dataset_unsqueezed = dataset_torch.unsqueeze(1).float()
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

print(X_test.shape)

model_DLPU = DLPU()

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

trained_model1,list_metric1,list_loss1 = model_train(
                                            model=model_DLPU,
                                            name='DLPU-GoogleCloud',
                                            batch_size=16,
                                            total_epochs=1000,
                                            learning_rate=0.0002,
                                            loss_freq=1,
                                            metric_freq=1,
                                            lr_freq=2,
                                            save_freq=10)
