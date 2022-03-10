from SSL import *
from pathlib import Path
from os.path import isfile, isdir, join
from os import listdir
import random
import matplotlib.pyplot as plt
from itertools import product

def get_possibilities(K):
  possibilities = list(product(list(range(K)), repeat = K))
  possibilities = torch.tensor(possibilities)
  return torch.nn.functional.one_hot(possibilities,K)


def SSO(model, X, possibilities, nb_bests = 5):
  '''
    Go throught primal then dual, compute dual for nb_bests label possibilities
    (most probable label according to primal) for querries 
    and keep the solution with less loss for support prediction on supports used as querries
    Parameters
    model: primal-dual model
    possibilities: all possible label predictions (calculated by get_predictions, to calculate only once)
    X: entry for primal and dual
    nb_bests: n 
    Returns predicted labels with and without SSO  
  '''

  primal = model.prime
  dual = model.dual
  N = model.n_support
  K = model.n_way
  
  assert nb_bests <= len(possibilities)
  assert possibilities.shape[1] == possibilities.shape[2] and possibilities.shape[1] == K
  # apply backbone
  X = model.feature(X.flatten(0,1))

  # apply linear transformation
  X = model.fc(X)

  X = X.view(K, -1, X.size(1))

  # reshape for primal
  X = torch.cat([torch.cat([torch.cat([X[:, :N], X[:, N + i: N + i + 1]], dim=1).view(1, -1, X.size(2)), model.labels], dim = 2) for i in range(model.n_query)], dim=0)

  
  # get labels
  first_labels = X[:,::N+1,-K:].argmax(dim = 2)

  # apply primal
  X_primal, W_primal = primal.forward(X)

  X_class = X_primal[:,:,-K:]

  X_querries_class = X_class[:,N::N+1] 


  X_querries_class = F.softmax(X_querries_class, dim = 2)

  # compute distance between possible labels and querry
  dist = ((X_querries_class.unsqueeze(1) - possibilities.unsqueeze(0))**2).sum(dim=(2,3))
  
  # take nb_best possibilities to test
  best_classes = dist.argsort(dim = 1)[:,:nb_bests] 
  

  best_classes = possibilities[best_classes].transpose(0,1) # size nb_bests x nb_querries x K x K

  new_X = torch.clone(X).repeat(nb_bests,1,1)

  # shift the last support as a query
  new_X[:,N-1::N+1] = X[:,N::(N + 1)].repeat(nb_bests,1,1)
 
  # shift the querry as the last support
  new_X[:, N::N+1] = X[:,N-1::N+1].repeat(nb_bests,1,1)

  # set the querry class as 0
  new_X[:,N::N+1,-K:] = 0
  # set labels for last support (previous querries)
  new_X[:,N-1::N+1, -K:] = best_classes.flatten(0,1)
  
  # apply dual
  X_dual, _ = dual(new_X)

  # calculate loss for querries (previous supports) => loss in form querry*nb_bests
  X_dual = ((X_dual[:,N::N+1,-K:].argmax(dim = 2) - first_labels.repeat(nb_bests,1))**2).sum(dim = 1)
  

  # get possibility with the less loss
  X_dual = X_dual.view((nb_bests,X_dual.shape[0]//nb_bests)).argmin(dim = 0)

  # get prediction for them
  Y_SSO = best_classes[X_dual,range(best_classes.shape[1])]

  # D = Y_SSO
  Y = best_classes[0,range(best_classes.shape[1])]
  
  return Y_SSO, Y


def testSSO(model, test_loader,nb_bests = 5):
    model.eval()
    acc_ = []
    acc_SSO = []
    conf_mat = torch.zeros((2,2)) # to see difference with and without SSO
    for i, (X,_ ) in enumerate(test_loader):
    
      labels = torch.arange(model.n_way).repeat(model.n_query)
      # labels = torch.from_numpy(np.repeat(range(model.n_way), model.n_query)) 
      

      labels = labels.to("cuda")
      # print(labels)
      
      X = X.to("cuda")
        
      with torch.no_grad():
        Y_HAT_SSO, Y_HAT = SSO(model, X, get_possibilities(model.n_way).cuda(), nb_bests = nb_bests)
        _, Y_HAT_SSO = torch.max(Y_HAT_SSO, dim = 2) 
        _ , Y_HAT = torch.max(Y_HAT, dim = 2) 

        # print(Y_HAT_SSO)
        # print(Y_HAT)
        # raise

        labels = labels.view(-1)
        Y_HAT_SSO = Y_HAT_SSO.view(-1)
        Y_HAT = Y_HAT.view(-1)
        
        correct_SSO = Y_HAT_SSO.eq(labels).cpu()
        correct = Y_HAT.eq(labels).cpu()    
        

        conf_mat[1][1] += torch.logical_and((correct == True),(correct_SSO == True)).sum()
        conf_mat[0][0] += torch.logical_and((correct == False),(correct_SSO == False)).sum()
        conf_mat[1][0] += torch.logical_and((correct == True),(correct_SSO == False)).sum()
        conf_mat[0][1] += torch.logical_and((correct == False),(correct_SSO == True)).sum()

        acc = correct.sum() / labels.numel()
        accSSO = correct_SSO.sum() / labels.numel()
        acc_.append(acc)
        acc_SSO.append(accSSO)
      

    print('avg_acc {:f}+/-{:f} with SSO K = 1'.format(torch.mean(torch.tensor(acc_,dtype = torch.float)).item(),torch.std(torch.tensor(acc_,dtype = torch.float)).item()) )
    print('avg_acc {:f}+/-{:f} with SSO'.format(torch.mean(torch.tensor(acc_SSO,dtype = torch.float)).item(),torch.std(torch.tensor(acc_SSO,dtype = torch.float)).item()) )
    conf_mat = conf_mat/torch.sum(conf_mat)
    return conf_mat

def getGraph(model, test_loader):
    # figure 5(b) de l'article
    model = model.cuda()
    model.eval()
    acc_SSO = []
    acc_ = []
    for i, (X,Y) in enumerate(test_loader):
      # labels = torch.from_numpy(np.repeat(range(model.n_way), model.n_query)) 
      labels = torch.arange(model.n_way).repeat(model.n_query)
      labels = labels.to("cuda")
      X = X.to("cuda")

      for nb_bests in range(1,21):
        with torch.no_grad():
          Y_HAT_SSO, _ = SSO(model, X, get_possibilities(model.n_way).cuda(), nb_bests = nb_bests)
          _, Y_HAT_SSO = torch.max(Y_HAT_SSO, dim = 2)  
          
          
          labels = labels.view(-1)
          
          Y_HAT_SSO = Y_HAT_SSO.view(-1)
        
          correct_SSO = Y_HAT_SSO.eq(labels).cpu()   
          accSSO = correct_SSO.sum() / labels.numel()
          acc_SSO.append(labels.numel() - correct_SSO.sum())
        
        
      plt.plot(range(1,21),acc_SSO)
      plt.xticks(range(2,21,2),range(2,21,2))
      plt.yticks(range(min(acc_SSO),max(acc_SSO) + 1),range(min(acc_SSO),max(acc_SSO) + 1))
      plt.show()
      break


def CUB_setup():

  data_path = '/content/cub/cub-200-dataset/'
  savedir = '/content/save_dir/'
  dataset = "CUB"

  folder_list = [f for f in listdir(data_path) if isdir(join(data_path, f))]
  print('{} dataset contains {} categories'.format("CUB", len(folder_list)))
  folder_list.sort()
  label_dict = dict(zip(folder_list,range(0,len(folder_list))))

  classfile_list_all = []

  for i, folder in enumerate(folder_list):
      folder_path = join(data_path, folder)
      classfile_list_all.append( [ join(folder_path, cf) for cf in listdir(folder_path) if (isfile(join(folder_path,cf)) and cf[0] != '.')])
      random.shuffle(classfile_list_all[i])

  file_list = []
  label_list = []
  for i, classfile_list in enumerate(classfile_list_all):
      file_list = file_list + classfile_list
      label_list = label_list + np.repeat(i, len(classfile_list)).tolist()

  fo = open(savedir + dataset + ".json", "w")
  fo.write('{"label_names": [')
  fo.writelines(['"%s",' % item  for item in folder_list])
  fo.seek(0, os.SEEK_END)
  fo.seek(fo.tell()-1, os.SEEK_SET)
  fo.write('],')

  fo.write('"image_names": [')
  fo.writelines(['"%s",' % item  for item in file_list])
  fo.seek(0, os.SEEK_END)
  fo.seek(fo.tell()-1, os.SEEK_SET)
  fo.write('],')

  fo.write('"image_labels": [')
  fo.writelines(['%d,' % item  for item in label_list])
  fo.seek(0, os.SEEK_END)
  fo.seek(fo.tell()-1, os.SEEK_SET)
  fo.write(']}')
  fo.close()
  print("CUB -OK")

def test(model, test_loader, mode = "Prime"):
    model.eval()
    acc_ = []
    for i, (x,_ ) in enumerate(test_loader):

      labels = torch.from_numpy(np.repeat(range(model.n_way), model.n_query))
      if torch.cuda.is_available():
        labels = labels.cuda()
      if mode == "Prime":
          prime_predicted_scores, dual_predicted_scores = model.forward_prime(x)
          y_pred_softmax = F.softmax(prime_predicted_scores, dim = 1)
          
      else:
          prime_predicted_scores, dual_predicted_scores = model.forward_dual(x)
          y_pred_softmax = F.softmax(dual_predicted_scores, dim = 1)
          

      _ , y_pred_tags = torch.max(y_pred_softmax, dim = 1) 
      
      labels = labels.view(-1)
      y_pred_tags = y_pred_tags.view(-1)
      
      correct = y_pred_tags.eq(labels)   
      acc = correct.sum() / labels.numel()
      acc_.append(acc)

    avg_acc = torch.mean(torch.tensor(acc_)).item()
    std_acc = torch.std(torch.tensor(acc_)).item()
    print('avg_acc {:f}'.format(avg_acc))
    return avg_acc, std_acc

if __name__ == '__main__':
  no_cub = False
  base_file = "save_dir/base.json"
  eval_file = "save_dir/val.json"
  novel_file = "save_dir/novel.json"
  base_file = "save_dir/CUB.json"

  model_path = "saved_models/model_dual_1.pch"
  n_shot = 5
  n_way = 5
  n_query = 16
  image_size = 224
  
  
  try:
    CUB_setup()
  except Exception as e:
    print("no cub detected")
    no_cub = True
  train_few_shot_params  = dict(n_way = n_way, n_support = n_shot) # n_support = n_shot

  novel_datamgr  = MyDataManager( image_size, n_query = n_query,  **train_few_shot_params)
  novel_loader = novel_datamgr.get_data_loader(novel_file , aug = True)
  if not no_cub:
    cub_datamgr  = MyDataManager( image_size, n_query = n_query,  **train_few_shot_params)
    cub_loader  = cub_datamgr.get_data_loader(base_file , aug = True)

  backbone = Backbone(type = 10)
  model = Model(n_way, n_shot, n_query, 512, backbone = backbone)
  model.load_state_dict(torch.load(Path(model_path)))
  
  
  
  

  if torch.cuda.is_available():
    print("Using GPU")
    model = model.cuda()

  #getGraph(model, novel_loader)
  #raise
  
  
  torch.manual_seed(1)
  mean_acc, std_acc = test(model,novel_loader)
  print("Without SSO:",mean_acc,"+/-",std_acc)
  

  torch.manual_seed(1)
  print(testSSO(model,novel_loader,nb_bests=5))
  
  if not no_cub:
    torch.manual_seed(1)
    print("CUB results")
    
    mean_acc, std_acc = test(model,cub_loader)
    print("Without SSO:",mean_acc,"+/-",std_acc)
    
    torch.manual_seed(1)
    print(testSSO(model,cub_loader))
  
  



