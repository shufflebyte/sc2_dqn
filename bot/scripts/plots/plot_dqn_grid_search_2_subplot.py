
# creates a subplot of grid search 1
# parameter: mini batch size (which stores the chosen experiences for training in DQN)
# values: 64, 256, 1024, 2048

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import pandas

import numpy as np

# data
results_tuple = [
    # neue Messreihe ab 1xx mit sample_size = 4 statt 32
    ('/Users/florian/ray_results/DQN102/DQN_srv_2019-07-01_07-21-38cstnun1y/progress.csv', ['train batch size = 256', 'gamma = 0.99']),
    ('/Users/florian/ray_results/DQN104/DQN_srv_2019-07-01_15-11-22km8iip_c/progress.csv', ['train batch size = 256', 'gamma = 0.99']),
    ('/Users/florian/ray_results/DQN106/DQN_srv_2019-07-01_22-14-12c96p8b19/progress.csv', ['train batch size = 256', 'gamma = 0.99']),
    ('/Users/florian/ray_results/DQN108/DQN_srv_2019-07-02_09-53-529uy879tx/progress.csv', ['train batch size = 256', 'gamma = 0.99']),
    ('/Users/florian/ray_results/DQN110/DQN_srv_2019-07-02_16-17-37o3benz4d/progress.csv', ['train batch size = 256', 'gamma = 0.99']),
    ('/Users/florian/ray_results/DQN112/DQN_srv_2019-07-02_23-43-527jbpxyva/progress.csv', ['train batch size = 256', 'gamma = 0.99']),
    ('/Users/florian/ray_results/DQN114/DQN_srv_2019-07-03_16-56-13zl3asmyn/progress.csv', ['train batch size = 256', 'gamma = 0.99']),
    ('/Users/florian/ray_results/DQN117/DQN_srv_2019-07-04_11-35-40rlcy6chm/progress.csv', ['train batch size = 256', 'gamma = 0.99']),
    ('/Users/florian/ray_results/DQN121/DQN_srv_2019-07-05_11-25-00gn42lc7v/progress.csv', ['train batch size = 256', 'gamma = 0.99']),
    ('/Users/florian/ray_results/DQN122/DQN_srv_2019-07-05_22-06-406w8wt3ob/progress.csv', ['train batch size = 256', 'gamma = 0.99']),
    ('/Users/florian/ray_results/DQN141/DQN_srv_2019-07-11_10-03-10x0vmx5hj/progress.csv', ['', 'gamma = 0.9']),
    ('/Users/florian/ray_results/DQN142/DQN_srv_2019-07-11_10-03-52vo60h17x/progress.csv', ['', 'gamma = 0.9']),
    ('/Users/florian/ray_results/DQN143/DQN_srv_2019-07-11_13-57-33mby4c0ks/progress.csv', ['', 'gamma = 0.9']),
    ('/Users/florian/ray_results/DQN144/DQN_srv_2019-07-11_13-58-29qq727hrp/progress.csv', ['', 'gamma = 0.9']),
    ('/Users/florian/ray_results/DQN145/DQN_srv_2019-07-11_17-08-15mkfd9tek/progress.csv', ['', 'gamma = 0.9']),
    ('/Users/florian/ray_results/DQN146/DQN_srv_2019-07-11_17-09-42cjlbegvw/progress.csv', ['', 'gamma = 0.9']),
    ('/Users/florian/ray_results/DQN147/DQN_srv_2019-07-12_11-49-40do1y17qt/progress.csv', ['', 'gamma = 0.9']),
    ('/Users/florian/ray_results/DQN148/DQN_srv_2019-07-12_11-50-44d1vxucjc/progress.csv', ['', 'gamma = 0.9']),
    ('/Users/florian/ray_results/DQN149/DQN_srv_2019-07-12_15-33-384v2e05c9/progress.csv', ['', 'gamma = 0.9']),
    ('/Users/florian/ray_results/DQN150/DQN_srv_2019-07-12_15-34-502dzr0_ui/progress.csv', ['', 'gamma = 0.9']),
    ('/Users/florian/ray_results/DQN151/DQN_srv_2019-07-12_21-43-20zjmi3w_d/progress.csv', ['', 'gamma = 0.8']),
    ('/Users/florian/ray_results/DQN152/DQN_srv_2019-07-12_21-44-24jav3ghjw/progress.csv', ['', 'gamma = 0.8']),
    ('/Users/florian/ray_results/DQN153/DQN_srv_2019-07-13_12-12-53a_hct11s/progress.csv', ['', 'gamma = 0.8']),
    ('/Users/florian/ray_results/DQN154/DQN_srv_2019-07-13_12-14-14w8p_94lr/progress.csv', ['', 'gamma = 0.8']),
    ('/Users/florian/ray_results/DQN155/DQN_srv_2019-07-14_09-44-19r2diknjy/progress.csv', ['', 'gamma = 0.8']),
    ('/Users/florian/ray_results/DQN156/DQN_srv_2019-07-14_09-45-24kywh6080/progress.csv', ['', 'gamma = 0.8']),
    ('/Users/florian/ray_results/DQN157/DQN_srv_2019-07-14_13-00-39gxmpvk38/progress.csv', ['', 'gamma = 0.8']),
    ('/Users/florian/ray_results/DQN158/DQN_srv_2019-07-14_13-01-47mcc97f8d/progress.csv', ['', 'gamma = 0.8']),
    ('/Users/florian/ray_results/DQN159/DQN_srv_2019-07-14_16-11-04z1qkfb_h/progress.csv', ['', 'gamma = 0.8']),
    ('/Users/florian/ray_results/DQN160/DQN_srv_2019-07-14_16-12-15l4_j6485/progress.csv', ['', 'gamma = 0.8'])
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
y = 0*x+147

# find the right column in label array by giving the right grid search number
grid_search_number = 1

#i does not refer to the arrays indexing, but to the DQN-ID, though i = 1 is DQN101 but i-1 is the array position

# create panda.DataFrames and bring them to subplots

fig = plt.figure()

plt.subplot(221)
for i in range(0, 9):
    data = pandas.read_csv(filepath_or_buffer=results_tuple[i][0], delimiter=',', header=1, usecols=[2, 20],
                           names=['reward_mean', 'seconds'])
    plt.plot(data['seconds'] / 60 / 60, data['reward_mean'], label='_nolegend_')
plt.title('gamma = 0.99')
plt.plot(x, y, linestyle='--', label='human reference')
plt.xlabel("time in hours")
plt.ylabel("mean reward")
# scale axes
x1,x2,y1,y2 = plt.axis()
plt.axis((x1, 3, 0, 180))
x1, x2 = plt.xlim()
plt.xlim([0, x2])
y1, y2 = plt.ylim()
plt.ylim([0, y2])

plt.subplot(222)
for i in range(10, 19):
    data = pandas.read_csv(filepath_or_buffer=results_tuple[i][0], delimiter=',', header=1, usecols=[2, 20],
                           names=['reward_mean', 'seconds'])
    plt.plot(data['seconds'] / 60 / 60, data['reward_mean'], label='_nolegend_')
plt.title('gamma = 0.9')
plt.plot(x, y, linestyle='--', label='human reference')
plt.xlabel("time in hours")
plt.ylabel("mean reward")
# scale axes
x1,x2,y1,y2 = plt.axis()
plt.axis((x1, 3, 0, 180))
x1, x2 = plt.xlim()
plt.xlim([0, x2])
y1, y2 = plt.ylim()
plt.ylim([0, y2])

plt.subplot(223)
for i in range(20, 29):
    data = pandas.read_csv(filepath_or_buffer=results_tuple[i][0], delimiter=',', header=1, usecols=[2, 20],
                           names=['reward_mean', 'seconds'])
    plt.plot(data['seconds'] / 60 / 60, data['reward_mean'], label='_nolegend_')
plt.title('gamma = 0.8')
plt.plot(x, y, linestyle='--', label='human reference')
plt.xlabel("time in hours")
plt.ylabel("mean reward")
# scale axes
x1,x2,y1,y2 = plt.axis()
plt.axis((x1, 3, 0, 180))
x1, x2 = plt.xlim()
plt.xlim([0, x2])
y1, y2 = plt.ylim()
plt.ylim([0, y2])

########
plt.grid()
plt.legend()
plt.show()
