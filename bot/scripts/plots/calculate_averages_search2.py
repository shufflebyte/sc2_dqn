# This file loads datasets from grid search 1 or 2 (change code) and processes it as follows:
# load data into dataFrame, insert uniformly distributed time steps
# interpolate over these time steps
# delete any NaN rows
# calculate a mean over all measurements
# plot the final curve/ curves

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
    {'file': '/Users/florian/ray_results/DQN140/DQN_srv_2019-07-09_17-05-18hrk3xm9i/progress.csv', 'label': {'search1': 'train batch size = 256', 'search2': 'gamma = 0.9'}},
    {'file': '/Users/florian/ray_results/DQN141/DQN_srv_2019-07-11_10-03-10x0vmx5hj/progress.csv', 'label': {'search1': 'train batch size = 256', 'search2': 'gamma = 0.9'}},
    {'file': '/Users/florian/ray_results/DQN142/DQN_srv_2019-07-11_10-03-52vo60h17x/progress.csv', 'label': {'search1': 'train batch size = 256', 'search2': 'gamma = 0.9'}},
    {'file': '/Users/florian/ray_results/DQN143/DQN_srv_2019-07-11_13-57-33mby4c0ks/progress.csv', 'label': {'search1': 'train batch size = 256', 'search2': 'gamma = 0.9'}},
    {'file': '/Users/florian/ray_results/DQN144/DQN_srv_2019-07-11_13-58-29qq727hrp/progress.csv', 'label': {'search1': 'train batch size = 256', 'search2': 'gamma = 0.9'}},
    {'file': '/Users/florian/ray_results/DQN145/DQN_srv_2019-07-11_17-08-15mkfd9tek/progress.csv', 'label': {'search1': 'train batch size = 256', 'search2': 'gamma = 0.9'}},
    {'file': '/Users/florian/ray_results/DQN146/DQN_srv_2019-07-11_17-09-42cjlbegvw/progress.csv', 'label': {'search1': 'train batch size = 256', 'search2': 'gamma = 0.9'}},
    {'file': '/Users/florian/ray_results/DQN147/DQN_srv_2019-07-12_11-49-40do1y17qt/progress.csv', 'label': {'search1': 'train batch size = 256', 'search2': 'gamma = 0.9'}},
    {'file': '/Users/florian/ray_results/DQN148/DQN_srv_2019-07-12_11-50-44d1vxucjc/progress.csv', 'label': {'search1': 'train batch size = 256', 'search2': 'gamma = 0.9'}},
    {'file': '/Users/florian/ray_results/DQN149/DQN_srv_2019-07-12_15-33-384v2e05c9/progress.csv', 'label': {'search1': 'train batch size = 256', 'search2': 'gamma = 0.9'}},
    {'file': '/Users/florian/ray_results/DQN150/DQN_srv_2019-07-12_15-34-502dzr0_ui/progress.csv', 'label': {'search1': 'train batch size = 256', 'search2': 'gamma = 0.9'}},
    {'file': '/Users/florian/ray_results/DQN151/DQN_srv_2019-07-12_21-43-20zjmi3w_d/progress.csv', 'label': {'search1': 'train batch size = 256', 'search2': 'gamma = 0.8', 'search3': "lr = 0.0001"}},
    {'file': '/Users/florian/ray_results/DQN152/DQN_srv_2019-07-12_21-44-24jav3ghjw/progress.csv', 'label': {'search1': 'train batch size = 256', 'search2': 'gamma = 0.8', 'search3': "lr = 0.0001"}},
    {'file': '/Users/florian/ray_results/DQN153/DQN_srv_2019-07-13_12-12-53a_hct11s/progress.csv', 'label': {'search1': 'train batch size = 256', 'search2': 'gamma = 0.8', 'search3': "lr = 0.0001"}},
    {'file': '/Users/florian/ray_results/DQN154/DQN_srv_2019-07-13_12-14-14w8p_94lr/progress.csv', 'label': {'search1': 'train batch size = 256', 'search2': 'gamma = 0.8', 'search3': "lr = 0.0001"}},
    {'file': '/Users/florian/ray_results/DQN155/DQN_srv_2019-07-14_09-44-19r2diknjy/progress.csv', 'label': {'search1': 'train batch size = 256', 'search2': 'gamma = 0.8', 'search3': "lr = 0.0001"}},
    {'file': '/Users/florian/ray_results/DQN156/DQN_srv_2019-07-14_09-45-24kywh6080/progress.csv', 'label': {'search1': 'train batch size = 256', 'search2': 'gamma = 0.8', 'search3': "lr = 0.0001"}},
    {'file': '/Users/florian/ray_results/DQN157/DQN_srv_2019-07-14_13-00-39gxmpvk38/progress.csv', 'label': {'search1': 'train batch size = 256', 'search2': 'gamma = 0.8', 'search3': "lr = 0.0001"}},
    {'file': '/Users/florian/ray_results/DQN158/DQN_srv_2019-07-14_13-01-47mcc97f8d/progress.csv', 'label': {'search1': 'train batch size = 256', 'search2': 'gamma = 0.8', 'search3': "lr = 0.0001"}},
    {'file': '/Users/florian/ray_results/DQN159/DQN_srv_2019-07-14_16-11-04z1qkfb_h/progress.csv', 'label': {'search1': 'train batch size = 256', 'search2': 'gamma = 0.8', 'search3': "lr = 0.0001"}},
    {'file': '/Users/florian/ray_results/DQN160/DQN_srv_2019-07-14_16-12-15l4_j6485/progress.csv', 'label': {'search1': 'train batch size = 256', 'search2': 'gamma = 0.8', 'search3': "lr = 0.0001"}},
]



# rewards_mean: 2
# rewards_max: 0
# iterations: 15
# timesteps: 13
# seconds: 20
# find the paths in ray_results:
# find . -name \progress.csv -print | grep DQN
# print(results[0]['file'], results[0]['label']['search1'])

fig = plt.figure()

# generate human expert data
x = np.linspace(0, 5, 100)
y = 0*x+147
plt.plot(x, y, linestyle='--', label='human reference')

# outer loop for calculating the averages of several curves
for x in range(0, 3):
    if x == 0:
        parameter = 'gamma = 0.99'
    if x == 1:
        parameter = 'gamma = 0.9'
    if x == 2:
        parameter = 'gamma = 0.8'

    # use counter for renaming columns in dataFrame
    i = 0

    # create empty df for all data
    df_all_data = pd.DataFrame()

    #inner loop for processing every TimeSeries
    for result in results:
        if 'search2' in result['label'] and result['label']['search2'] == parameter:
            print("Processing dataset ", i)

            # read it
            df = pd.read_csv(filepath_or_buffer=result['file'],
                             delimiter=',',
                             header=1,
                             usecols=[2, 20],
                             names=['mean'+str(i), 'S', ]
                             )

            # get the latest timestamp (last row, column "S" where the seconds of progress.csv are stored
            n = int(df['S'].iloc[-1])

            # since the smallest time steps in the measurements are 2.5 to 3 seconds and the
            # largest time steps are something like 30 seconds, an sampling with 1 second
            # large time steps will be more than enough

            # in the following the oversampling_factor is multiplied the number of measurements
            # to create an over/undersampling
            oversampling_factor = 1
            sample_points = np.linspace(0, n, oversampling_factor*n+1)

            # create a df with only sample points, call it only one letter, because there fucks something up..
            df_sample_points = pd.DataFrame(sample_points, columns=list('S'))

            # append new index column: Therefore ignore indexing (because they will not be unique, also do sorting
            # to make it be sorted :-p , also because the standard behavior will be changed and will be unsorted in
            # future versions of pandas..
            df_sampled = df.append(df_sample_points, ignore_index=True, sort=True)
            df_sampled = df_sampled.sort_values(by="S")
            # after sorting, interpolate the NaNs between the real sampling points
            df_sampled = df_sampled.interpolate(method='linear', axis=0)
            # let S be the index (for concatenation AND for plotting)
            df_sampled = df_sampled.set_index("S")
            # concat the new timeSeries with the others
            df_all_data = pd.concat([df_all_data, df_sampled], axis=1, sort=True, join='outer')
        # increase i for the naming of the columns with the Y-Values
        i += 1
    # delete "any" NaNs, that are left and do it inplace (do not make a copy of dataFrame)
    df_all_data.dropna(how='any', inplace=True)
    # calculate the mean for the 10 series and create a mean column, but skip non-numerical and NaNs
    df_all_data['mean'] = df_all_data.mean(numeric_only=True, skipna=True, axis=1, )

    # add the mean to the plot! Afterwards df_all_data will be deleted and replaced by new data!
    plt.plot(df_all_data.index.values/60/60, df_all_data['mean'], label=parameter)
    # print(df_all_data)

# Now modify Matplotlib for a consistens scaling and view
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, 3, 0, 180))
x1, x2 = plt.xlim()
plt.xlim([0, x2])
y1, y2 = plt.ylim()
plt.ylim([0, y2])

plt.xlabel("time in hours")
plt.ylabel("mean reward")

plt.legend()
plt.show()
