# Plot results from csv file

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import pandas

import numpy as np

results_tuple = [
    ('/Users/florian/ray_results/DQN001/DQN_srv_2019-06-15_09-01-53bvolt_5a/progress.csv', ["train batch size = 1024"]), # 0
    ('/Users/florian/ray_results/DQN002/DQN_srv_2019-06-15_09-01-50dw98p535/progress.csv', ["train batch size = 2048"]),
    ('/Users/florian/ray_results/DQN003/DQN_srv_2019-06-14_18-02-00ok73e6of/progress.csv', ["train batch size = 256", "", "gamma = 0.99"]),
    ('/Users/florian/ray_results/DQN004/DQN_srv_2019-06-14_18-05-44rghse78i/progress.csv', ['train batch size = 64']),
    ('/Users/florian/ray_results/DQN005/DQN_srv_2019-06-15_15-45-46uchnfm6j/progress.csv', ['', 'train batch size = 1024']),
    ('/Users/florian/ray_results/DQN006/DQN_srv_2019-06-15_18-51-52aoq4225l/progress.csv', ['', 'train batch size = 2048']),
    ('/Users/florian/ray_results/DQN007/DQN_srv_2019-06-15_22-26-494h_a9eh_/progress.csv', ['', 'train batch size = 256']),
    ('/Users/florian/ray_results/DQN008/DQN_srv_2019-06-16_10-06-41vfhlp6gl/progress.csv', ['', 'train batch size = 64']),
    ('/Users/florian/ray_results/DQN009/DQN_srv_2019-06-16_10-47-284cxbc9ag/progress.csv', ['','', 'gamma = 0.9']),
    ('/Users/florian/ray_results/DQN010/DQN_srv_2019-06-16_15-08-27a01btwsy/progress.csv', ['','', 'gamma = 0.8']),
    ('/Users/florian/ray_results/DQN011/DQN_srv_2019-06-16_15-21-24udyj1n07/progress.csv', ['','', 'lr = 0.001']), # 10
    ('/Users/florian/ray_results/DQN012/DQN_srv_2019-06-16_19-43-30tuphlj0i/progress.csv', ['','', 'lr = 0.01']),
    ('/Users/florian/ray_results/DQN013/DQN_srv_2019-06-28_08-59-50nuj6gx4n/progress.csv', ["train batch size = 2048"]),
    ('/Users/florian/ray_results/DQN014/DQN_srv_2019-06-28_09-03-316uxxabjf/progress.csv', ["train batch size = 1024"]),
    ('/Users/florian/ray_results/DQN015/DQN_srv_2019-06-29_10-17-44rzkfe9cx/progress.csv', ["train batch size = 256"]),
    ('/Users/florian/ray_results/DQN016/DQN_srv_2019-06-29_10-18-49j6qreny8/progress.csv', ["train batch size = 64"]),
    ('/Users/florian/ray_results/DQN017/DQN_srv_2019-06-28_11-40-42pdmjlzx2/progress.csv', ["train batch size = 2048"]),
    ('/Users/florian/ray_results/DQN018/DQN_srv_2019-06-28_11-42-22qhf5ufwp/progress.csv', ["train batch size = 1024"]),
    ('/Users/florian/ray_results/DQN019/DQN_srv_2019-06-29_20-13-33g5us4dxa/progress.csv', ["train batch size = 256"]),
    ('/Users/florian/ray_results/DQN020/DQN_srv_2019-06-29_20-14-59yoz221wu/progress.csv', ["train batch size = 64"]),
    # neue Messreihe ab 10x mit sample_size = 4 statt 32
    ('/Users/florian/ray_results/DQN101/DQN_srv_2019-07-01_07-15-44thfl65bi/progress.csv', ['train batch size = 64']), #20
    ('/Users/florian/ray_results/DQN102/DQN_srv_2019-07-01_07-21-38cstnun1y/progress.csv', ['train batch size = 256']),
    ('/Users/florian/ray_results/DQN103/DQN_srv_2019-07-01_15-10-40woak50mq/progress.csv', ['train batch size = 64']),
    ('/Users/florian/ray_results/DQN104/DQN_srv_2019-07-01_15-11-22km8iip_c/progress.csv', ['train batch size = 256']),
    ('/Users/florian/ray_results/DQN105/DQN_srv_2019-07-01_22-13-295skioave/progress.csv', ['train batch size = 64']),
    ('/Users/florian/ray_results/DQN106/DQN_srv_2019-07-01_22-14-12c96p8b19/progress.csv', ['train batch size = 256']),
    ('/Users/florian/ray_results/DQN107/DQN_srv_2019-07-02_09-52-48_22cz_lj/progress.csv', ['train batch size = 64']),
    ('/Users/florian/ray_results/DQN108/DQN_srv_2019-07-02_09-53-529uy879tx/progress.csv', ['train batch size = 256']),
    ('/Users/florian/ray_results/DQN109/DQN_srv_2019-07-02_16-16-12x3k84ohv/progress.csv', ['train batch size = 64']),
    ('/Users/florian/ray_results/DQN110/DQN_srv_2019-07-02_16-17-37o3benz4d/progress.csv', ['train batch size = 256']),
    ('/Users/florian/ray_results/DQN111/DQN_srv_2019-07-02_23-42-48tgw8hxkh/progress.csv', ['train batch size = 64']), #30
    ('/Users/florian/ray_results/DQN112/DQN_srv_2019-07-02_23-43-527jbpxyva/progress.csv', ['train batch size = 256']),
    ('/Users/florian/ray_results/DQN113/DQN_srv_2019-07-03_12-10-19nhsu9cso/progress.csv', ['train batch size = 64']),
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
#x = np.linspace(0, 5, 100)
#y = 0*x+147


# make a figure:

# make multiple plots alongside in on figure
fig = plt.figure()
# find the right column in label array by giving the right grid search number
grid_search_number = 1

# fetch data into pandas.dataFrame and hand it over to pyplot object!
for i in [20, 22, 24, 26, 28, 30, 32]: # 64
#for i in [21, 23, 25, 27, 29, 31]:  # 256
    data = pandas.read_csv(filepath_or_buffer=results_tuple[i][0], delimiter=',', header=1, usecols=[2, 20],
                           names=['reward_mean', 'seconds'])
    plt.plot(data['seconds'] / 60 / 60, data['reward_mean'], label=results_tuple[i][1][grid_search_number - 1])

# add human expert to plot
#plt.plot(x, y, linestyle='--', label='human reference')

# make some modificatin to the pyplot object
# get axis stuff
x1,x2,y1,y2 = plt.axis()
# set axis stuff
plt.axis((x1, 5, 0, 150))

# make plots start at the most left and most bottom...
x1, x2 = plt.xlim()
plt.xlim([0, x2])

y1, y2 = plt.ylim()
plt.ylim([0, y2])
# set extra ticks
#plt.yticks(list(plt.yticks()[0]) + [150])

plt.legend()
plt.show()