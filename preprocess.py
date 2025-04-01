import numpy as np
from os import path
import mne

####################################################################################################
# Path and channels setting
path_RawData   = './data/ISRUC_S3/RawData/'
path_output    = './data/ISRUC_S3/'
channels = ['C3_A2', 'C4_A1', 'F3_A2', 'F4_A1', 'O1_A2', 'O2_A1', 'LOC_A2', 'ROC_A1','X1', 'X2']
####################################################################################################


def read_psg(path_RawData, sub_id, channels, resample_freq=100, ignore=30):
    channels = [c.replace("_", "-") for c in channels]
    data = mne.io.read_raw_edf(path.join(path_RawData, '%d'%sub_id, '%d.edf'%sub_id), preload=True)

    data.notch_filter(50) # 50Hz notch filter
    data.filter(0.3, 50) # 0.3-50Hz bandpass filter

    data.resample(resample_freq) # resample to 100Hz

    data_select = data.get_data(picks=channels)
    data_select_re = data_select.reshape(10, -1, 3000).transpose(1, 0, 2)
    data_select_final = data_select_re[:-ignore] # ignore last 30 epochs

    print(data_select.shape, data_select_re.shape, data_select_final.shape)
    return data_select_final*1e6


def read_label(path_RawData, sub_id, ignore=30):
    label = []
    with open(path.join(path_RawData, '%d/%d_1.txt' % (sub_id, sub_id))) as f:
        s = f.readline()
        while True:
            a = s.replace('\n', '')
            label.append(int(a))
            s = f.readline()
            if s == '' or s == '\n':
                break
    return np.array(label[:-ignore])


####################################################################################################
'''
output:
    save to $path_output/ISRUC_S3.npz:
        Fold_data:  [k-fold] list, each element is [N,V,T]
        Fold_label: [k-fold] list, each element is [N,C] (one-hot)
        Fold_len:   [k-fold] list
    N: number of samples
    V: number of channels
    T: number of time points
    C: number of classes
'''

fold_label = []
fold_psg = []
fold_len = []

for sub in range(1, 11):
    print('#'*80)
    print('Read subject', sub)
    label = read_label(path_RawData, sub)
    psg = read_psg(path_RawData, sub, channels)
    print('Subject', sub, ':', label.shape, psg.shape)
    assert len(label) == len(psg)

    # in ISRUC, 0-Wake, 1-N1, 2-N2, 3-N3, 5-REM
    label[label==5] = 4  # make 4 correspond to REM
    fold_label.append(np.eye(5)[label])
    fold_psg.append(psg)
    fold_len.append(len(label))
print('Preprocess over.')

np.savez(path.join(path_output, 'ISRUC_S3.npz'),
    Fold_data = np.array(fold_psg, dtype=object),
    Fold_label = np.array(fold_label, dtype=object),
    Fold_len = fold_len
)
print('Saved to', path.join(path_output, 'ISRUC_S3.npz'))
