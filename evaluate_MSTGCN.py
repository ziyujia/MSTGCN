import os
import numpy as np
import argparse
import shutil
import gc
import time
import torch.nn as nn

from model.Dataset import SimpleDataset, TwoOutputDataset
from model.FeatureNet import FeatureNet
from model.MSTGCN import MSTGCN
from model.DataGenerator import kFoldGenerator_test, DomainGenerator_test
from model.Utils import *

print(128 * '#')
print('Start to evaluate MSTGCN.')

# # 1. Get configuration

# ## 1.1. Read .config file

# command line parameters -c -g
parser = argparse.ArgumentParser()
parser.add_argument("-c", type = str, help = "configuration file", required = True)
parser.add_argument("-g", type = str, help = "GPU number to use", required = True)
args = parser.parse_args()
Path, cfgFeature, cfgTrain, cfgModel = ReadConfig(args.c)

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
lambda_GRL = float(cfgTrain["lambda_GRL"])

# [train] parameters ('_f' means FeatureNet)
channels   = int(cfgFeature["channels"])
fold       = int(cfgFeature["fold"])
num_epochs_f = int(cfgFeature["epoch_f"])
batch_size_f = int(cfgFeature["batch_size_f"])
optimizer_f  = cfgFeature["optimizer_f"]
learn_rate_f = float(cfgFeature["learn_rate_f"])

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
Fold_Data  = ReadList['Fold_data']   # Data of each fold
Fold_Label = ReadList['Fold_label']  # Labels of each fold

print("Read data successfully")
print('Number of samples: ',np.sum(Fold_Num))

# ## 2.2. Read adjacency matrix
# Prepare Chebyshev polynomial of G_DC
Dis_Conn = np.load(Path['disM'], allow_pickle=True)  # shape:[V,V]
L_DC = scaled_Laplacian(Dis_Conn)                    # Calculate laplacian matrix
cheb_poly_DC = cheb_polynomial(L_DC, cheb_k)         # K-order Chebyshev polynomial

print("Read data successfully")
Fold_Num_c  = Fold_Num + 1 - context
print('Number of samples: ',np.sum(Fold_Num), '(with context:', np.sum(Fold_Num_c), ')')

# ## 2.3. Build kFoldGenerator or DomainGenerator
Data_Generator = kFoldGenerator_test(Fold_Data, Fold_Label)
Dom_Generator = DomainGenerator_test(Fold_Num_c)


# # 3. Model training (cross validation)

# k-fold cross validation
all_scores = []
for i in range(fold):
    print(128*'_')
    print('Fold #', i)
    print(time.asctime(time.localtime(time.time())))
    
    # get i th-fold data
    test_data, test_targets = Data_Generator.getFold(i)
    test_domain = Dom_Generator.getFold(i)

    test_data = np.float32(test_data)
    teDataset = SimpleDataset(test_data, test_targets)
    teGen = DataLoader(teDataset,
                       batch_size = batch_size_f,
                       shuffle = False,
                       num_workers = 0)

    ## build FeatureNet
    model_feature = FeatureNet(channels).cuda()
    loss_func = nn.CrossEntropyLoss()

    # load the weights of best performance
    model_feature.eval()
    model_feature.load_state_dict(torch.load(Path['Save']+'FeatureNet_Best_'+str(i)+'.pth'))

    # Get feature
    test_feature = get_feature_dataset(model_feature, teDataset, batch_size_f)
    
    ## Use the feature to evaluate MSTGCN
    print('Feature',test_feature.shape)
    print(Fold_Num)
    test_feature, test_targets    = AddContext_SingleSub(test_feature, test_targets, context)
    print('Feature with context:',test_feature.shape, test_targets.shape)
    
    test_feature = np.float32(test_feature)
    teDataset = TwoOutputDataset(test_feature, test_targets, test_domain)
    teGen = DataLoader(teDataset,
                       batch_size = batch_size,
                       shuffle = False,
                       num_workers = 0)
    
    ## build MSTGCN
    model = MSTGCN(context, channels, 256, cheb_k, num_of_chev_filters, num_of_time_filters,
                   time_conv_strides, cheb_poly_DC, time_conv_kernel, num_block, dense_size, 
                   GLalpha,  dropout, lambda_GRL, num_classes=5, num_domain=fold-1).cuda()
    loss_func = nn.CrossEntropyLoss()
    loss_domain = nn.CrossEntropyLoss()

    # load the weights of best performance
    model.eval()
    model.load_state_dict(torch.load(Path['Save']+'MSTGCN_Best_'+str(i)+'.pth'))
    
    print('[TEST]')
    acc, loss, loss_c, loss_d, test_pred, test_true = val_MSTGCN(model, teGen, loss_func, loss_domain, True)
    print()
    
    all_scores.append(acc)
    
    if i == 0:
        AllPred = test_pred
        AllTrue = test_true
    else:
        AllPred = np.concatenate((AllPred, test_pred))
        AllTrue = np.concatenate((AllTrue, test_true))
    
    del model, test_feature, test_targets
    gc.collect()

# # 4. Final results

# print acc of each fold
print(128*'=')
print("All folds' acc: ",all_scores)
print("Average acc of each fold: ",np.mean(all_scores))

# Print score to console
print(128*'=')
PrintScore(AllTrue, AllPred)
# Print score to Result.txt file
PrintScore(AllTrue, AllPred, savePath=Path['Save'], savefile='Result_test.txt')

# Print confusion matrix and save
ConfusionMatrix(AllTrue, AllPred, classes=['W','N1','N2','N3','REM'], savePath=Path['Save'])

print('End of evaluating MSTGCN.')
print(128 * '#')
