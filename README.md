# MSTGCN

Multi-View Spatial-Temporal Graph Convolutional Networks with Domain Generalization for Sleep Stage Classification. [[paper](https://ieeexplore.ieee.org/document/9530406)]

> This work is an extension of previous work: *GraphSleepNet: Adaptive Spatial-Temporal Graph Convolutional Networks for Sleep Stage Classification* (IJCAI 2020) [[paper](https://www.ijcai.org/proceedings/2020/184)].

![model_architecture](fig/MSTGCN.png)

These are source code and experimental setup for the ISRUC-S3 dataset.

> ***Note:** The code is based on the **PyTorch** now.*

## Citation

If you find this useful, please cite our work as follows:

```latex
@ARTICLE{9530406,
  author={Jia, Ziyu and Lin, Youfang and Wang, Jing and Ning, Xiaojun and He, Yuanlai and Zhou, Ronghao and Zhou, Yuhan and Lehman, Li-wei H.},
  journal={IEEE Transactions on Neural Systems and Rehabilitation Engineering}, 
  title={Multi-View Spatial-Temporal Graph Convolutional Networks With Domain Generalization for Sleep Stage Classification}, 
  year={2021},
  volume={29},
  number={},
  pages={1977-1986},
  doi={10.1109/TNSRE.2021.3110665}}
```


## Datasets

We evaluate our model on the ISRUC-Sleep-S3 dataset and the Montreal Archive of Sleep Studies (MASS)-SS3 dataset.

- The **ISRUC-Sleep-S3** dataset is available [here](https://sleeptight.isr.uc.pt/), and we provide the pipeline to run MSTGCN on it.
- The **MASS-SS3** dataset is an open-access and collaborative database of laboratory-based polysomnography (PSG) recordings. Information on how to obtain it can be found [here](http://massdb.herokuapp.com/en/).

## Verified environment

- Python 3.10
- CUDA 12.1
- cuDNN 8.9.2
- PyTorch 2.1.2
- scipy 1.14.1
- scikit-learn 1.5.1
- mne 1.5.0

## How to run

- **1. Get Dataset:**
  
  You can download ISRUC-Sleep-S3 dataset by the following command, which will automatically download the raw data and extracted data to `./data/ISRUC_S3/`:

  ```shell
  bash get_ISRUC_S3.sh
  ```

  For MASS-SS3 dataset, you need to you need to visit the [link](http://massdb.herokuapp.com/en/) and contact their administrator.

- **2. Data preparation:**

  To facilitate reading, we preprocess the dataset into a single .npz file:

  ```shell
  python preprocess.py
  ```
  
  In addition, distance based adjacency matrix is provided at `./data/ISRUC_S3/DistanceMatrix.npy`.
  
- **3. Configuration:**

  Write the config file in the format of the example.

  We provide a config file at `/config/ISRUC_S3.config`

- **4. Feature extraction:**

  Run `python train_FeatureNet.py` with -c and -g parameters. After this step, the features learned by a feature net will be stored.

  + -c: The configuration file.
  + -g: The number of the GPU to use. E.g.,`0`,`1,3`. 

  ```shell
  python train_FeatureNet.py -c ./config/ISRUC_S3.config -g 0
  ```

- **5. Train MSTGCN:**

  Run `python train_MSTGCN.py` with -c and -g parameters. This step uses the extracted features directly. 

    ```shell
  python train_MSTGCN.py -c ./config/ISRUC_S3.config -g 0
    ```

- **6. Evaluate MSTGCN:**

  Run `python evaluate_MSTGCN.py` with -c and -g parameters.

    ```shell
  python evaluate_MSTGCN.py -c ./config/ISRUC_S3.config -g 0
    ```


> **Summary of commands to run:**
>
> ```shell
> bash get_ISRUC_S3.sh
> python preprocess.py
> python train_FeatureNet.py -c ./config/ISRUC_S3.config -g 0
> python train_MSTGCN.py -c ./config/ISRUC_S3.config -g 0
> python evaluate_MSTGCN.py -c ./config/ISRUC_S3.config -g 0
> ```
>

