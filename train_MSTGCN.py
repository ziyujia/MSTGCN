import os
import numpy as np
import argparse
import shutil
import gc

from model.MSTGCN import MSTGCN
from model.DataGenerator import DomainGenerator_train
from model.Utils import *

import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from model.Dataset import TwoOutputDataset

print(128 * '#')
print('Start to train MSTGCN.')

# # 1. Get configuration

# ## 1.1. Read .config file

# command line parameters -c -g
parser = argparse.ArgumentParser()
parser.add_argument("-c", type = str, help = "configuration file", required = True)
parser.add_argument("-g", type = str, help = "GPU number to use", required = True)
args = parser.parse_args()
Path, _, cfgTrain, cfgModel = ReadConfig(args.c)

# set GPU number
os.environ["CUDA_VISIBLE_DEVICES"] = args.g


# ## 1.2. Analytic parameters

# [train] parameters ('_f' means FeatureNet)
channels   = int(cfgTrain["channels"])
fold       = int(cfgTrain["fold"])
context    = int(cfgTrain["context"])
num_epochs = int(cfgTrain["epoch"])
batch_size = int(cfgTrain["batch_size"])
optimizer  = cfgTrain["optimizer"]
learn_rate = float(cfgTrain["learn_rate"])
lambda_GRL   = float(cfgTrain["lambda_GRL"])

# [model] parameters
dense_size            = np.array(str.split(cfgModel["Globaldense"],','),dtype=int)
GLalpha               = float(cfgModel["GLalpha"])
num_of_chev_filters   = int(cfgModel["cheb_filters"])
num_of_time_filters   = int(cfgModel["time_filters"])
time_conv_strides     = int(cfgModel["time_conv_strides"])
time_conv_kernel      = int(cfgModel["time_conv_kernel"])
num_block             = int(cfgModel["num_block"])
cheb_k                = int(cfgModel["cheb_k"])
l2                    = float(cfgModel["l2"])
dropout               = float(cfgModel["dropout"])

# ## 1.3. Parameter check and enable

# Create save pathand copy .config to it
if not os.path.exists(Path['Save']):
    os.makedirs(Path['Save'])
shutil.copyfile(args.c, Path['Save']+"last.config")


# # 2. Read data and process data

# ## 2.1. Read data
# Each fold corresponds to one subject's data (ISRUC-S3 dataset)
ReadList = np.load(Path['data'], allow_pickle=True)
Fold_Num   = ReadList['Fold_len']    # Num of samples of each fold

# ## 2.2. Read adjacency matrix
# Prepare Chebyshev polynomial of G_DC
Dis_Conn = np.load(Path['disM'], allow_pickle=True)  # shape:[V,V]
L_DC = scaled_Laplacian(Dis_Conn)                    # Calculate laplacian matrix
cheb_poly_DC = cheb_polynomial(L_DC, cheb_k)         # K-order Chebyshev polynomial

print("Read data successfully")
Fold_Num_c  = Fold_Num + 1 - context
print('Number of samples: ',np.sum(Fold_Num), '(with context:', np.sum(Fold_Num_c), ')')

# ## 2.3. Build kFoldGenerator or DomainGenerator
Dom_Generator = DomainGenerator_train(Fold_Num_c)


# # 3. Model training (cross validation)

# k-fold cross validation
all_scores = []
fit_loss = []
fit_acc = []
fit_val_loss = []
fit_val_acc = []

def train_MSTGCN_fold(i):
    print(128*'_')
    print('Fold #', i)
    print(time.asctime(time.localtime(time.time())))
    
    # get i th-fold data
    Features = np.load(Path['Save']+'Feature_'+str(i)+'.npz', allow_pickle=True)
    train_feature = np.float32(Features['train_feature'])
    train_targets = Features['train_targets']
    val_feature   = np.float32(Features['val_feature'])
    val_targets   = Features['val_targets']
    
    ## Use the feature to train MSTGCN
    print('Feature',train_feature.shape,val_feature.shape)
    train_feature, train_targets  = AddContext_MultiSub(train_feature, train_targets,
                                                        np.delete(Fold_Num.copy(), [i,(i+9)%10]), context, i)
    val_feature, val_targets      = AddContext_SingleSub(val_feature, val_targets, context)
    print('Feature with context:',train_feature.shape, val_feature.shape)
    
    train_domain, val_domain = Dom_Generator.getFold(i)
    
    trDataset = TwoOutputDataset(np.float32(train_feature), train_targets, train_domain)
    cvDataset = TwoOutputDataset(np.float32(val_feature), val_targets, val_domain)
    trGen = DataLoader(trDataset,
                       batch_size = batch_size,
                       shuffle = True,
                       num_workers = 0)
    cvGen = DataLoader(cvDataset,
                       batch_size = batch_size,
                       shuffle = False,
                       num_workers = 0)
    
    ## build MSTGCN & train
    model = MSTGCN(context, channels, 256, cheb_k, num_of_chev_filters, num_of_time_filters,
                   time_conv_strides, cheb_poly_DC, time_conv_kernel, num_block, dense_size, 
                   GLalpha,  dropout, lambda_GRL, num_classes=5, num_domain=fold-1).cuda()
    loss_func = nn.CrossEntropyLoss(weight = torch.FloatTensor([1,1.2,1,1,1.25]).cuda())
    loss_domain = nn.CrossEntropyLoss()
    opt = Instantiation_optim(optimizer, learn_rate, model, l2) # optimizer of FeatureNet
    
    best_acc = 0
    count_epoch = 0
    tr_acc_list_e = []
    tr_loss_list_e = []
    tr_loss_c_list_e = []
    tr_loss_d_list_e = []
    val_acc_list_e = []
    val_loss_list_e = []
    val_loss_c_list_e = []
    val_loss_d_list_e = []
    for epoch in range(num_epochs):
        time_start = time.time()
        
        tr_acc, tr_loss, tr_loss_c, tr_loss_d = train_epoch_MSTGCN(model, trGen, loss_func, loss_domain, opt, epoch)
        va_acc, va_loss, va_loss_c, va_loss_d = val_MSTGCN(model, cvGen, loss_func, loss_domain, epoch, False)

        # Save training information
        tr_acc_list_e.append(tr_acc)
        tr_loss_list_e.append(tr_loss)
        tr_loss_c_list_e.append(tr_loss_c)
        tr_loss_d_list_e.append(tr_loss_d)
        val_acc_list_e.append(va_acc)
        val_loss_list_e.append(va_loss)
        val_loss_c_list_e.append(va_loss_c)
        val_loss_d_list_e.append(va_loss_d)

        # Save best & Early stopping
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), Path['Save']+'MSTGCN_Best_'+str(i)+'.pth')
            print(" U ", end='')
            count_epoch = 0
        else:
            count_epoch += 1
            print("   ", end='')
            if count_epoch >= 20:
                print(" ES ")
                break
        time_end = time.time()
        time_cost = time_end - time_start
        print(" Time:%.3f" % (time_cost))

    # load the weights of best performance
    model.eval()
    model.load_state_dict(torch.load(Path['Save']+'MSTGCN_Best_'+str(i)+'.pth'))
    
    saveFile = open(Path['Save'] + "Result_MSTGCN.txt", 'a+')
    print('Fold #'+str(i), file=saveFile)
    print('TR_ACC:', tr_acc_list_e, '; TR_loss:', tr_loss_list_e, '; TR_loss_c:', tr_loss_c_list_e, '; TR_loss_d:', tr_loss_d_list_e,
          '; VA_ACC:', val_acc_list_e, '; VA_loss:', val_loss_list_e, '; VA_loss_c:', val_loss_c_list_e, '; VA_loss_d:', val_loss_d_list_e,
          file=saveFile)
    saveFile.close()

    del model, train_targets, val_targets, train_feature, val_feature
    gc.collect()
    
    return tr_loss_list_e, tr_acc_list_e, val_loss_list_e, val_acc_list_e


# # 4. Collect training results

for i in range(fold):
    fit_loss_i, fit_acc_i, fit_val_loss_i, fit_val_acc_i = train_MSTGCN_fold(i)
    
    VariationCurve(fit_acc_i, fit_val_acc_i, 'Acc_MSTGCN_'+str(i), Path['Save'], figsize=(9, 6))
    VariationCurve(fit_loss_i, fit_val_loss_i, 'Loss_MSTGCN_'+str(i), Path['Save'], figsize=(9, 6))
    
    fit_acc.append(fit_acc_i)
    fit_val_loss.append(fit_val_loss_i)
    fit_val_acc.append(fit_val_acc_i)
    fit_loss.append(fit_loss_i)
    

print(128 * '_')
print('End of training MSTGCN.')
print(128 * '#')
