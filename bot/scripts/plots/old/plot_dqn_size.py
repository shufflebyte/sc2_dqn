# Plot results from csv file

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import pandas

import numpy as np

# results_tuple = [
#     ('/Users/florian/ray_results/DQN001/DQN_srv_2019-06-15_09-01-53bvolt_5a/progress.csv', ["train batch size = 1024"]),  # 0
#     ('/Users/florian/ray_results/DQN002/DQN_srv_2019-06-15_09-01-50dw98p535/progress.csv', ["train batch size = 2048"]),
#     ('/Users/florian/ray_results/DQN003/DQN_srv_2019-06-14_18-02-00ok73e6of/progress.csv', ["train batch size = 256", "", "gamma = 0.99"]),
#     ('/Users/florian/ray_results/DQN004/DQN_srv_2019-06-14_18-05-44rghse78i/progress.csv', ['train batch size = 64']),
#     ('/Users/florian/ray_results/DQN005/DQN_srv_2019-06-15_15-45-46uchnfm6j/progress.csv', ['', 'train batch size = 1024']),
#     ('/Users/florian/ray_results/DQN006/DQN_srv_2019-06-15_18-51-52aoq4225l/progress.csv', ['', 'train batch size = 2048']),
#     ('/Users/florian/ray_results/DQN007/DQN_srv_2019-06-15_22-26-494h_a9eh_/progress.csv', ['', 'train batch size = 256']),
#     ('/Users/florian/ray_results/DQN008/DQN_srv_2019-06-16_10-06-41vfhlp6gl/progress.csv', ['', 'train batch size = 64']),
#     ('/Users/florian/ray_results/DQN009/DQN_srv_2019-06-16_10-47-284cxbc9ag/progress.csv', ['', '', 'gamma = 0.9']),
#     ('/Users/florian/ray_results/DQN010/DQN_srv_2019-06-16_15-08-27a01btwsy/progress.csv', ['', '', 'gamma = 0.8']),
#     ('/Users/florian/ray_results/DQN011/DQN_srv_2019-06-16_15-21-24udyj1n07/progress.csv', ['', '', 'lr = 0.001']),  # 10
#     ('/Users/florian/ray_results/DQN012/DQN_srv_2019-06-16_19-43-30tuphlj0i/progress.csv', ['', '', 'lr = 0.01']),
#     ('/Users/florian/ray_results/DQN013/DQN_srv_2019-06-28_08-59-50nuj6gx4n/progress.csv', ["train batch size = 2048"]),
#     ('/Users/florian/ray_results/DQN014/DQN_srv_2019-06-28_09-03-316uxxabjf/progress.csv', ["train batch size = 1024"]),
#     ('/Users/florian/ray_results/DQN015/DQN_srv_2019-06-29_10-17-44rzkfe9cx/progress.csv', ["train batch size = 256"]),
#     ('/Users/florian/ray_results/DQN016/DQN_srv_2019-06-29_10-18-49j6qreny8/progress.csv', ["train batch size = 64"]),
#     ('/Users/florian/ray_results/DQN017/DQN_srv_2019-06-28_11-40-42pdmjlzx2/progress.csv', ["train batch size = 2048"]),
#     ('/Users/florian/ray_results/DQN018/DQN_srv_2019-06-28_11-42-22qhf5ufwp/progress.csv', ["train batch size = 1024"]),
#     ('/Users/florian/ray_results/DQN019/DQN_srv_2019-06-29_20-13-33g5us4dxa/progress.csv', ["train batch size = 256"]),
#     ('/Users/florian/ray_results/DQN020/DQN_srv_2019-06-29_20-14-59yoz221wu/progress.csv', ["train batch size = 64"]),
# ]


results_tuple = [
    # neue Messreihe ab 1xx mit sample_size = 4 statt 32
    ('/Users/florian/ray_results/DQN101/DQN_srv_2019-07-01_07-15-44thfl65bi/progress.csv', ['train batch size = 64']),
    ('/Users/florian/ray_results/DQN102/DQN_srv_2019-07-01_07-21-38cstnun1y/progress.csv', ['train batch size = 256']),
    ('/Users/florian/ray_results/DQN103/DQN_srv_2019-07-01_15-10-40woak50mq/progress.csv', ['train batch size = 64']),
    ('/Users/florian/ray_results/DQN104/DQN_srv_2019-07-01_15-11-22km8iip_c/progress.csv', ['train batch size = 256']),
    ('/Users/florian/ray_results/DQN105/DQN_srv_2019-07-01_22-13-295skioave/progress.csv', ['train batch size = 64']),
    ('/Users/florian/ray_results/DQN106/DQN_srv_2019-07-01_22-14-12c96p8b19/progress.csv', ['train batch size = 256']),
    ('/Users/florian/ray_results/DQN107/DQN_srv_2019-07-02_09-52-48_22cz_lj/progress.csv', ['train batch size = 64']),
    ('/Users/florian/ray_results/DQN108/DQN_srv_2019-07-02_09-53-529uy879tx/progress.csv', ['train batch size = 256']),
    ('/Users/florian/ray_results/DQN109/DQN_srv_2019-07-02_16-16-12x3k84ohv/progress.csv', ['train batch size = 64']),
    ('/Users/florian/ray_results/DQN110/DQN_srv_2019-07-02_16-17-37o3benz4d/progress.csv', ['train batch size = 256']),
    ('/Users/florian/ray_results/DQN111/DQN_srv_2019-07-02_23-42-48tgw8hxkh/progress.csv', ['train batch size = 64']),
    ('/Users/florian/ray_results/DQN112/DQN_srv_2019-07-02_23-43-527jbpxyva/progress.csv', ['train batch size = 256']),
    ('/Users/florian/ray_results/DQN113/DQN_srv_2019-07-03_12-10-19nhsu9cso/progress.csv', ['train batch size = 64']),
    ('/Users/florian/ray_results/DQN114/DQN_srv_2019-07-03_16-56-13zl3asmyn/progress.csv', ['train batch size = 256']),
    ('/Users/florian/ray_results/DQN115/DQN_srv_2019-07-03_20-35-481607xz0k/progress.csv', ['train batch size = 64']),
    ('/Users/florian/ray_results/DQN116/DQN_srv_2019-07-04_11-34-356e_dbzbn/progress.csv', ['train batch size = 64']),
    ('/Users/florian/ray_results/DQN117/DQN_srv_2019-07-04_11-35-40rlcy6chm/progress.csv', ['train batch size = 256']),
    ('/Users/florian/ray_results/DQN118/DQN_srv_2019-07-04_20-26-16n3ccsfk2/progress.csv', ['train batch size = 2048']),
    ('/Users/florian/ray_results/DQN119/DQN_srv_2019-07-04_20-28-32l2o_700b/progress.csv', ['train batch size = 2048']),
    ('/Users/florian/ray_results/DQN120/DQN_srv_2019-07-05_11-23-099v6szmyu/progress.csv', ['train batch size = 64']),
    ('/Users/florian/ray_results/DQN121/DQN_srv_2019-07-05_11-25-00gn42lc7v/progress.csv', ['train batch size = 256']),
    ('/Users/florian/ray_results/DQN122/DQN_srv_2019-07-05_22-06-406w8wt3ob/progress.csv', ['train batch size = 256']),
    ('/Users/florian/ray_results/DQN123/DQN_srv_2019-07-05_22-08-48fasrvdho/progress.csv', ['train batch size = 2048']),
    ('/Users/florian/ray_results/DQN124/DQN_srv_2019-07-06_10-27-39mki1r8r4/progress.csv', ['train batch size = 1024']),
    ('/Users/florian/ray_results/DQN125/DQN_srv_2019-07-06_10-28-39yfo0vzxb/progress.csv', ['train batch size = 2048']),
    ('/Users/florian/ray_results/DQN126/DQN_srv_2019-07-06_15-51-56bp3a7qua/progress.csv', ['train batch size = 1024']),
    ('/Users/florian/ray_results/DQN127/DQN_srv_2019-07-06_15-56-4859p1ktmg/progress.csv', ['train batch size = 2048']),
    ('/Users/florian/ray_results/DQN128/DQN_srv_2019-07-07_11-55-41fxx0e1xx/progress.csv', ['train batch size = 1024']),
    ('/Users/florian/ray_results/DQN129/DQN_srv_2019-07-07_11-57-008a4hcje7/progress.csv', ['train batch size = 2048']),
    ('/Users/florian/ray_results/DQN130/DQN_srv_2019-07-07_18-47-33idsmlk4i/progress.csv', ['train batch size = 1024']),
    ('/Users/florian/ray_results/DQN131/DQN_srv_2019-07-07_18-51-0400m90kqo/progress.csv', ['train batch size = 2048']),
    ('/Users/florian/ray_results/DQN132/DQN_srv_2019-07-08_09-42-37p57ueswl/progress.csv', ['train batch size = 1024']),
    ('/Users/florian/ray_results/DQN133/DQN_srv_2019-07-08_09-45-3924ap6ue_/progress.csv', ['train batch size = 2048']),
    ('/Users/florian/ray_results/DQN134/DQN_srv_2019-07-08_15-03-480ce90twm/progress.csv', ['train batch size = 1024']),
    ('/Users/florian/ray_results/DQN135/DQN_srv_2019-07-08_15-05-19e6gbbd1m/progress.csv', ['train batch size = 2048']),
    ('/Users/florian/ray_results/DQN136/DQN_srv_2019-07-08_20-35-034wup5px4/progress.csv', ['train batch size = 1024']),
    ('/Users/florian/ray_results/DQN137/DQN_srv_2019-07-08_20-36-502kfb58z4/progress.csv', ['train batch size = 2048']),
    ('/Users/florian/ray_results/DQN138/DQN_srv_2019-07-09_09-27-12phaa6_wy/progress.csv', ['train batch size = 1024']),
    ('/Users/florian/ray_results/DQN139/DQN_srv_2019-07-09_09-29-15je3wp6l_/progress.csv', ['train batch size = 1024']),
    ('/Users/florian/ray_results/DQN140/DQN_srv_2019-07-09_17-05-18hrk3xm9i/progress.csv', ['train batch size = 1024'])
]

# (path, [labels])

# find the paths in ray_results:
# find . -name \progress.csv -print | grep DQN

# the columns indices for the information:
# rewards_mean: 2
# rewards_max: 0
# iterations: 15
# timesteps: 13
# seconds: 20

# generate human expert data
x = np.linspace(0, 5, 100)
y = 0*x+147


# make a figure:

# make multiple plots alongside in on figure
fig = plt.figure()

plt.title('Measurements with blabla')
plt.xlabel('[training time] = hours')
plt.ylabel('mean reward')
# find the right column in label array by giving the right grid search number
grid_search_number = 1

# i refers to DQNi not to array i, thus every i-1 for labels and so on
# fetch data into pandas.dataFrame and hand it over to pyplot object!
#for i in [1, 3, 5, 7, 9, 11, 13, 15, 16, 20]: # 64, fertig
#for i in [2, 4, 6, 8, 10, 12, 14, 17, 21, 22]:  # 256, fertig
#for i in [24, 26, 28, 30, 32, 34, 36, 38, 39]: #, 40]: # 1024, fertig
for i in [18, 19, 23, 25, 27, 29, 31, 33, 35, 37]: # 2048, fertig
    data = pandas.read_csv(filepath_or_buffer=results_tuple[i-1][0], delimiter=',', header=1, usecols=[2, 20],
                           names=['reward_mean', 'seconds'])
    # plt.plot(data['seconds'] / 60 / 60, data['reward_mean'], label=results_tuple[i-1][1][grid_search_number - 1])
    plt.plot(data['seconds'] / 60 / 60, data['reward_mean'], label='_nolegend_')

# add human expert to plot
plt.plot(x, y, linestyle='--', label='human reference')

# make some modificatin to the pyplot object
# get axis stuff
x1,x2,y1,y2 = plt.axis()
# set axis stuff
plt.axis((x1, 5, 0, 180))

# make plots start at the most left and most bottom...
x1, x2 = plt.xlim()
plt.xlim([0, x2])

y1, y2 = plt.ylim()
plt.ylim([0, y2])

# set extra ticks
#plt.yticks(list(plt.yticks()[0]) + [150])

plt.legend()
plt.show()