import os
import numpy as np
import argparse
import shutil
import gc

from model.FeatureNet import FeatureNet
from model.DataGenerator import kFoldGenerator_train
from model.Utils import *

import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from model.Dataset import SimpleDataset

print(128 * '#')
print('Start to train FeatureNet.')

# # 1. Get configuration

# ## 1.1. Read .config file

# command line parameters -c -g
parser = argparse.ArgumentParser()
parser.add_argument("-c", type = str, help = "configuration file", required = True)
parser.add_argument("-g", type = str, help = "GPU number to use", required = True)
args = parser.parse_args()
Path, cfgFeature, _, _ = ReadConfig(args.c)

# set GPU number
os.environ["CUDA_VISIBLE_DEVICES"] = args.g

# ## 1.2. Analytic parameters

# [train] parameters ('_f' means FeatureNet)
channels   = int(cfgFeature["channels"])
fold       = int(cfgFeature["fold"])
num_epochs_f = int(cfgFeature["epoch_f"])
batch_size_f = int(cfgFeature["batch_size_f"])
optimizer_f  = cfgFeature["optimizer_f"]
learn_rate_f = float(cfgFeature["learn_rate_f"])


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
Fold_Data  = ReadList['Fold_data']   # Data of each fold
Fold_Label = ReadList['Fold_label']  # Labels of each fold

print("Read data successfully")
print('Number of samples: ',np.sum(Fold_Num))

# ## 2.2. Build kFoldGenerator or DomainGenerator
DataGenerator = kFoldGenerator_train(Fold_Data, Fold_Label)


# # 3. Model training (cross validation)

# k-fold cross validation
all_scores = []
fit_loss = []
fit_acc = []
fit_val_loss = []
fit_val_acc = []

def train_featurenet_fold(i):
    print(128*'_')
    print('Fold #', i)
    print(time.asctime(time.localtime(time.time())))
    
    # get i th-fold data
    train_data, train_targets, val_data, val_targets = DataGenerator.getFold(i)
    train_data = np.float32(train_data)
    val_data = np.float32(val_data)
    
    trDataset = SimpleDataset(train_data, train_targets)
    cvDataset = SimpleDataset(val_data, val_targets)
    trGen = DataLoader(trDataset,
                       batch_size = batch_size_f,
                       shuffle = True,
                       num_workers = 0)
    cvGen = DataLoader(cvDataset,
                       batch_size = batch_size_f,
                       shuffle = False,
                       num_workers = 0)
    
    ## build FeatureNet & train
    model = FeatureNet(channels).cuda()
    loss_func = nn.CrossEntropyLoss(weight = torch.FloatTensor([1,1.2,1,1,1.25]).cuda())  # [1,1.8,1,1.2,1.25] [1,1.5,1,1,1.5]
    opt_f = Instantiation_optim(optimizer_f, learn_rate_f, model, 0) # optimizer of FeatureNet
    
    best_acc = 0
    count_epoch = 0
    tr_acc_list_e = []
    tr_loss_list_e = []
    val_acc_list_e = []
    val_loss_list_e = []
    for epoch in range(num_epochs_f):
        time_start = time.time()
        
        tr_acc, tr_loss = train_epoch(model, trGen, loss_func, opt_f, epoch)
        va_acc, va_loss = val(model, cvGen, loss_func, epoch, False)

        # Save training information
        tr_acc_list_e.append(tr_acc)
        tr_loss_list_e.append(tr_loss)
        val_acc_list_e.append(va_acc)
        val_loss_list_e.append(va_loss)

        # Save best & Early stopping
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save(model.state_dict(), Path['Save']+'FeatureNet_Best_'+str(i)+'.pth')
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
    model.load_state_dict(torch.load(Path['Save']+'FeatureNet_Best_'+str(i)+'.pth'))
    
    # get and save the learned feature
    train_feature = get_feature_dataset(model, trDataset, batch_size_f)
    val_feature = get_feature_dataset(model, cvDataset, batch_size_f)

    print('Save feature of Fold #' + str(i) + ' to' + Path['Save']+'Feature_'+str(i) + '.npz')
    np.savez(Path['Save']+'Feature_'+str(i)+'.npz', 
        train_feature = train_feature,
        val_feature = val_feature,
        train_targets = train_targets,
        val_targets = val_targets
    )
    
    saveFile = open(Path['Save'] + "Result_FeatureNet.txt", 'a+')
    print('Fold #'+str(i), file=saveFile)
    print('TR_ACC:', tr_acc_list_e, '; TR_loss:', tr_loss_list_e, '; VA_ACC:', val_acc_list_e, '; VA_loss:', val_loss_list_e, file=saveFile)
    saveFile.close()

    del model, train_data, train_targets, val_data, val_targets, train_feature, val_feature
    gc.collect()
    
    return tr_loss_list_e, tr_acc_list_e, val_loss_list_e, val_acc_list_e


# # 4. Collect training results

for i in range(fold):
    fit_loss_i, fit_acc_i, fit_val_loss_i, fit_val_acc_i = train_featurenet_fold(i)
    
    VariationCurve(fit_acc_i, fit_val_acc_i, 'Acc_Feature_'+str(i), Path['Save'], figsize=(9, 6))
    VariationCurve(fit_loss_i, fit_val_loss_i, 'Loss_Feature_'+str(i), Path['Save'], figsize=(9, 6))
    
    fit_acc.append(fit_acc_i)
    fit_val_loss.append(fit_val_loss_i)
    fit_val_acc.append(fit_val_acc_i)
    fit_loss.append(fit_loss_i)
        
print(128 * '_')

print('End of training FeatureNet.')
print(128 * '#')
