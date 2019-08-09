
# creates a subplot of grid search 1
# parameter: mini batch size (which stores the chosen experiences for training in DQN)
# values: 64, 256, 1024, 2048

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import pandas as pd

import numpy as np

results = [
    {'file': '/Users/florian/ray_results/DQN101/DQN_srv_2019-07-01_07-15-44thfl65bi/progress.csv', 'label': {'search1': 'train batch size = 64'}},
    {'file': '/Users/florian/ray_results/DQN102/DQN_srv_2019-07-01_07-21-38cstnun1y/progress.csv', 'label': {'search1': 'train batch size = 256', 'search2': 'gamma = 0.99'}},
    {'file': '/Users/florian/ray_results/DQN103/DQN_srv_2019-07-01_15-10-40woak50mq/progress.csv', 'label': {'search1': 'train batch size = 64'}},
    {'file': '/Users/florian/ray_results/DQN104/DQN_srv_2019-07-01_15-11-22km8iip_c/progress.csv', 'label': {'search1': 'train batch size = 256', 'search2': 'gamma = 0.99'}},
    {'file': '/Users/florian/ray_results/DQN105/DQN_srv_2019-07-01_22-13-295skioave/progress.csv', 'label': {'search1': 'train batch size = 64'}},
    {'file': '/Users/florian/ray_results/DQN106/DQN_srv_2019-07-01_22-14-12c96p8b19/progress.csv', 'label': {'search1': 'train batch size = 256', 'search2': 'gamma = 0.99'}},
    {'file': '/Users/florian/ray_results/DQN107/DQN_srv_2019-07-02_09-52-48_22cz_lj/progress.csv', 'label': {'search1': 'train batch size = 64'}},
    {'file': '/Users/florian/ray_results/DQN108/DQN_srv_2019-07-02_09-53-529uy879tx/progress.csv', 'label': {'search1': 'train batch size = 256', 'search2': 'gamma = 0.99'}},
    {'file': '/Users/florian/ray_results/DQN109/DQN_srv_2019-07-02_16-16-12x3k84ohv/progress.csv', 'label': {'search1': 'train batch size = 64'}},
    {'file': '/Users/florian/ray_results/DQN110/DQN_srv_2019-07-02_16-17-37o3benz4d/progress.csv', 'label': {'search1': 'train batch size = 256', 'search2': 'gamma = 0.99'}},
    {'file': '/Users/florian/ray_results/DQN111/DQN_srv_2019-07-02_23-42-48tgw8hxkh/progress.csv', 'label': {'search1': 'train batch size = 64'}},
    {'file': '/Users/florian/ray_results/DQN112/DQN_srv_2019-07-02_23-43-527jbpxyva/progress.csv', 'label': {'search1': 'train batch size = 256', 'search2': 'gamma = 0.99'}},
    {'file': '/Users/florian/ray_results/DQN113/DQN_srv_2019-07-03_12-10-19nhsu9cso/progress.csv', 'label': {'search1': 'train batch size = 64'}},
    {'file': '/Users/florian/ray_results/DQN114/DQN_srv_2019-07-03_16-56-13zl3asmyn/progress.csv', 'label': {'search1': 'train batch size = 256', 'search2': 'gamma = 0.99'}},
    {'file': '/Users/florian/ray_results/DQN115/DQN_srv_2019-07-03_20-35-481607xz0k/progress.csv', 'label': {'search1': 'train batch size = 64'}},
    {'file': '/Users/florian/ray_results/DQN116/DQN_srv_2019-07-04_11-34-356e_dbzbn/progress.csv', 'label': {'search1': 'train batch size = 64'}},
    {'file': '/Users/florian/ray_results/DQN117/DQN_srv_2019-07-04_11-35-40rlcy6chm/progress.csv', 'label': {'search1': 'train batch size = 256', 'search2': 'gamma = 0.99'}},
    {'file': '/Users/florian/ray_results/DQN118/DQN_srv_2019-07-04_20-26-16n3ccsfk2/progress.csv', 'label': {'search1': 'train batch size = 2048'}},
    {'file': '/Users/florian/ray_results/DQN119/DQN_srv_2019-07-04_20-28-32l2o_700b/progress.csv', 'label': {'search1': 'train batch size = 2048'}},
    {'file': '/Users/florian/ray_results/DQN120/DQN_srv_2019-07-05_11-23-099v6szmyu/progress.csv', 'label': {'search1': 'train batch size = 64'}},
    {'file': '/Users/florian/ray_results/DQN121/DQN_srv_2019-07-05_11-25-00gn42lc7v/progress.csv', 'label': {'search1': 'train batch size = 256', 'search2': 'gamma = 0.99'}},
    {'file': '/Users/florian/ray_results/DQN122/DQN_srv_2019-07-05_22-06-406w8wt3ob/progress.csv', 'label': {'search1': 'train batch size = 256', 'search2': 'gamma = 0.99'}},
    {'file': '/Users/florian/ray_results/DQN123/DQN_srv_2019-07-05_22-08-48fasrvdho/progress.csv', 'label': {'search1': 'train batch size = 2048'}},
    {'file': '/Users/florian/ray_results/DQN124/DQN_srv_2019-07-06_10-27-39mki1r8r4/progress.csv', 'label': {'search1': 'train batch size = 1024'}},
    {'file': '/Users/florian/ray_results/DQN125/DQN_srv_2019-07-06_10-28-39yfo0vzxb/progress.csv', 'label': {'search1': 'train batch size = 2048'}},
    {'file': '/Users/florian/ray_results/DQN126/DQN_srv_2019-07-06_15-51-56bp3a7qua/progress.csv', 'label': {'search1': 'train batch size = 1024'}},
    {'file': '/Users/florian/ray_results/DQN127/DQN_srv_2019-07-06_15-56-4859p1ktmg/progress.csv', 'label': {'search1': 'train batch size = 2048'}},
    {'file': '/Users/florian/ray_results/DQN128/DQN_srv_2019-07-07_11-55-41fxx0e1xx/progress.csv', 'label': {'search1': 'train batch size = 1024'}},
    {'file': '/Users/florian/ray_results/DQN129/DQN_srv_2019-07-07_11-57-008a4hcje7/progress.csv', 'label': {'search1': 'train batch size = 2048'}},
    {'file': '/Users/florian/ray_results/DQN130/DQN_srv_2019-07-07_18-47-33idsmlk4i/progress.csv', 'label': {'search1': 'train batch size = 1024'}},
    {'file': '/Users/florian/ray_results/DQN131/DQN_srv_2019-07-07_18-51-0400m90kqo/progress.csv', 'label': {'search1': 'train batch size = 2048'}},
    {'file': '/Users/florian/ray_results/DQN132/DQN_srv_2019-07-08_09-42-37p57ueswl/progress.csv', 'label': {'search1': 'train batch size = 1024'}},
    {'file': '/Users/florian/ray_results/DQN133/DQN_srv_2019-07-08_09-45-3924ap6ue_/progress.csv', 'label': {'search1': 'train batch size = 2048'}},
    {'file': '/Users/florian/ray_results/DQN134/DQN_srv_2019-07-08_15-03-480ce90twm/progress.csv', 'label': {'search1': 'train batch size = 1024'}},
    {'file': '/Users/florian/ray_results/DQN135/DQN_srv_2019-07-08_15-05-19e6gbbd1m/progress.csv', 'label': {'search1': 'train batch size = 2048'}},
    {'file': '/Users/florian/ray_results/DQN136/DQN_srv_2019-07-08_20-35-034wup5px4/progress.csv', 'label': {'search1': 'train batch size = 1024'}},
    {'file': '/Users/florian/ray_results/DQN137/DQN_srv_2019-07-08_20-36-502kfb58z4/progress.csv', 'label': {'search1': 'train batch size = 2048'}},
    {'file': '/Users/florian/ray_results/DQN138/DQN_srv_2019-07-09_09-27-12phaa6_wy/progress.csv', 'label': {'search1': 'train batch size = 1024'}},
    {'file': '/Users/florian/ray_results/DQN139/DQN_srv_2019-07-09_09-29-15je3wp6l_/progress.csv', 'label': {'search1': 'train batch size = 1024'}},
    {'file': '/Users/florian/ray_results/DQN140/DQN_srv_2019-07-09_17-05-18hrk3xm9i/progress.csv', 'label': {'search1': 'train batch size = 1024'}},
    {'file': '/Users/florian/ray_results/DQN141/DQN_srv_2019-07-11_10-03-10x0vmx5hj/progress.csv', 'label': {'search2': 'gamma = 0.9'}},
    {'file': '/Users/florian/ray_results/DQN142/DQN_srv_2019-07-11_10-03-52vo60h17x/progress.csv', 'label': {'search2': 'gamma = 0.9'}},
    {'file': '/Users/florian/ray_results/DQN143/DQN_srv_2019-07-11_13-57-33mby4c0ks/progress.csv', 'label': {'search2': 'gamma = 0.9'}},
    {'file': '/Users/florian/ray_results/DQN144/DQN_srv_2019-07-11_13-58-29qq727hrp/progress.csv', 'label': {'search2': 'gamma = 0.9'}},
    {'file': '/Users/florian/ray_results/DQN145/DQN_srv_2019-07-11_17-08-15mkfd9tek/progress.csv', 'label': {'search2': 'gamma = 0.9'}},
    {'file': '/Users/florian/ray_results/DQN146/DQN_srv_2019-07-11_17-09-42cjlbegvw/progress.csv', 'label': {'search2': 'gamma = 0.9'}},
    {'file': '/Users/florian/ray_results/DQN147/DQN_srv_2019-07-12_11-49-40do1y17qt/progress.csv', 'label': {'search2': 'gamma = 0.9'}},
    {'file': '/Users/florian/ray_results/DQN148/DQN_srv_2019-07-12_11-50-44d1vxucjc/progress.csv', 'label': {'search2': 'gamma = 0.9'}},
    {'file': '/Users/florian/ray_results/DQN149/DQN_srv_2019-07-12_15-33-384v2e05c9/progress.csv', 'label': {'search2': 'gamma = 0.9'}},
    {'file': '/Users/florian/ray_results/DQN150/DQN_srv_2019-07-12_15-34-502dzr0_ui/progress.csv', 'label': {'search2': 'gamma = 0.9'}},
    {'file': '/Users/florian/ray_results/DQN151/DQN_srv_2019-07-12_21-43-20zjmi3w_d/progress.csv', 'label': {'search2': 'gamma = 0.8', 'search3': "lr = 0.0001"}},
    {'file': '/Users/florian/ray_results/DQN152/DQN_srv_2019-07-12_21-44-24jav3ghjw/progress.csv', 'label': {'search2': 'gamma = 0.8', 'search3': "lr = 0.0001"}},
    {'file': '/Users/florian/ray_results/DQN153/DQN_srv_2019-07-13_12-12-53a_hct11s/progress.csv', 'label': {'search2': 'gamma = 0.8', 'search3': "lr = 0.0001"}},
    {'file': '/Users/florian/ray_results/DQN154/DQN_srv_2019-07-13_12-14-14w8p_94lr/progress.csv', 'label': {'search2': 'gamma = 0.8', 'search3': "lr = 0.0001"}},
    {'file': '/Users/florian/ray_results/DQN155/DQN_srv_2019-07-14_09-44-19r2diknjy/progress.csv', 'label': {'search2': 'gamma = 0.8', 'search3': "lr = 0.0001"}},
    {'file': '/Users/florian/ray_results/DQN156/DQN_srv_2019-07-14_09-45-24kywh6080/progress.csv', 'label': {'search2': 'gamma = 0.8', 'search3': "lr = 0.0001"}},
    {'file': '/Users/florian/ray_results/DQN157/DQN_srv_2019-07-14_13-00-39gxmpvk38/progress.csv', 'label': {'search2': 'gamma = 0.8', 'search3': "lr = 0.0001"}},
    {'file': '/Users/florian/ray_results/DQN158/DQN_srv_2019-07-14_13-01-47mcc97f8d/progress.csv', 'label': {'search2': 'gamma = 0.8', 'search3': "lr = 0.0001"}},
    {'file': '/Users/florian/ray_results/DQN159/DQN_srv_2019-07-14_16-11-04z1qkfb_h/progress.csv', 'label': {'search2': 'gamma = 0.8', 'search3': "lr = 0.0001"}},
    {'file': '/Users/florian/ray_results/DQN160/DQN_srv_2019-07-14_16-12-15l4_j6485/progress.csv', 'label': {'search2': 'gamma = 0.8', 'search3': "lr = 0.0001"}},
    {'file': '/Users/florian/ray_results/DQN161/DQN_srv_2019-07-30_13-20-24n72_o9a7/progress.csv', 'label': {'search3': "lr = 0.001"}},
    {'file': '/Users/florian/ray_results/DQN162/DQN_srv_2019-07-30_13-21-58io5ws5xn/progress.csv', 'label': {'search3': "lr = 0.001"}},
    {'file': '/Users/florian/ray_results/DQN163/DQN_srv_2019-07-30_16-34-41boj29s2i/progress.csv', 'label': {'search3': "lr = 0.001"}},
    {'file': '/Users/florian/ray_results/DQN164/DQN_srv_2019-07-30_16-36-555a2rer8m/progress.csv', 'label': {'search3': "lr = 0.001"}},
    {'file': '/Users/florian/ray_results/DQN165/DQN_srv_2019-07-30_20-49-12z2pv57bo/progress.csv', 'label': {'search3': "lr = 0.001"}},
    {'file': '/Users/florian/ray_results/DQN166/DQN_srv_2019-07-30_20-50-481zr3cs_j/progress.csv', 'label': {'search3': "lr = 0.001"}},
    {'file': '/Users/florian/ray_results/DQN167/DQN_srv_2019-07-31_00-06-569migwwb_/progress.csv', 'label': {'search3': "lr = 0.001"}},
    {'file': '/Users/florian/ray_results/DQN168/DQN_srv_2019-07-31_00-08-33vi8cgi96/progress.csv', 'label': {'search3': "lr = 0.001"}},
    {'file': '/Users/florian/ray_results/DQN169/DQN_srv_2019-07-31_07-52-16gwjbb_ij/progress.csv', 'label': {'search3': "lr = 0.001"}},
    {'file': '/Users/florian/ray_results/DQN170/DQN_srv_2019-07-31_07-53-22wez0u5cd/progress.csv', 'label': {'search3': "lr = 0.001"}},
    {'file': '/Users/florian/ray_results/DQN171/DQN_srv_2019-08-03_12-05-55az7up74h/progress.csv', 'label': {'search3': "lr = 0.01"}},
    {'file': '/Users/florian/ray_results/DQN172/DQN_srv_2019-08-03_12-06-37jkzs_qvx/progress.csv', 'label': {'search3': "lr = 0.01"}},
    {'file': '/Users/florian/ray_results/DQN173/DQN_srv_2019-08-03_18-50-5675pq7r8f/progress.csv', 'label': {'search3': "lr = 0.01"}},
    {'file': '/Users/florian/ray_results/DQN174/DQN_srv_2019-08-03_18-52-00caok_kob/progress.csv', 'label': {'search3': "lr = 0.01"}},
    {'file': '/Users/florian/ray_results/DQN175/DQN_srv_2019-08-04_00-01-18l96sa1rm/progress.csv', 'label': {'search3': "lr = 0.01"}},
    {'file': '/Users/florian/ray_results/DQN176/DQN_srv_2019-08-04_00-02-498bs_ofv8/progress.csv', 'label': {'search3': "lr = 0.01"}},
    {'file': '/Users/florian/ray_results/DQN177/DQN_srv_2019-08-04_08-47-50q2cvpiqh/progress.csv', 'label': {'search3': "lr = 0.01"}},
    {'file': '/Users/florian/ray_results/DQN178/DQN_srv_2019-08-04_08-49-17sa52da31/progress.csv', 'label': {'search3': "lr = 0.01"}},
    {'file': '/Users/florian/ray_results/DQN179/DQN_srv_2019-08-04_12-37-12qikscbn0/progress.csv', 'label': {'search3': "lr = 0.01"}},
    {'file': '/Users/florian/ray_results/DQN180/DQN_srv_2019-08-04_12-38-31yt7xnx5l/progress.csv', 'label': {'search3': "lr = 0.01"}},
]

# the columns indices for the information:
# rewards_mean: 2
# rewards_max: 0
# iterations: 15
# timesteps: 13
# seconds: 20

# find the paths in ray_results:
# find . -name \progress.csv -print | grep DQN

# generate human expert data
x = np.linspace(0, 5, 100)
y = 0 * x + 147

# outer loop for calculating the averages of several curves
for p in range(0, 4):
    if p == 0:
        parameter = 'train batch size = 64'
    if p == 1:
        parameter = 'train batch size = 256'
    if p == 2:
        parameter = 'train batch size = 1024'
    if p == 3:
        parameter = 'train batch size = 2048'

    # use counter for renaming columns in dataFrame
    i = 0

    fig = plt.figure()

    #inner loop for processing every TimeSeries
    for result in results:
        if 'search1' in result['label'] and result['label']['search1'] == parameter:
            print("Processing dataset ", i)

            # read it
            df = pd.read_csv(filepath_or_buffer=result['file'],
                             delimiter=',',
                             header=1,
                             usecols=[2, 20],
                             names=['reward_mean', 'seconds']
                             )
            print(df)

            plt.plot(df['seconds'] / 60 / 60, df['reward_mean'], label='_nolegend_')

    plt.plot(x, y, linestyle='--', label='human reference')

    # Now modify Matplotlib for a consistent scaling and view
    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, 3, 0, 180))
    x1, x2 = plt.xlim()
    plt.xlim([0, x2])
    y1, y2 = plt.ylim()
    plt.ylim([0, y2])

    plt.title(parameter)
    plt.xlabel("time in hours")
    plt.ylabel("mean reward")

    plt.legend()
    plt.show()