# Plot results from csv file

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import pandas
path_ray = '/Users/florian/ray_results/'
files = ['DQN002/DQN_srv_2019-06-15_09-01-50dw98p535/progress.csv',
         'DQN001/DQN_srv_2019-06-15_09-01-53bvolt_5a/progress.csv',
         'DQN003/DQN_srv_2019-06-14_18-02-00ok73e6of/progress.csv',
         'DQN004/DQN_srv_2019-06-14_18-05-44rghse78i/progress.csv',
         'DQN005/DQN_srv_2019-06-15_15-45-46uchnfm6j/progress.csv',
         'DQN006/DQN_srv_2019-06-15_18-51-52aoq4225l/progress.csv',
         'DQN007/DQN_srv_2019-06-15_22-26-494h_a9eh_/progress.csv',
         'DQN008/DQN_srv_2019-06-16_10-06-41vfhlp6gl/progress.csv',
         'DQN009/DQN_srv_2019-06-16_10-47-284cxbc9ag/progress.csv',
         'DQN010/DQN_srv_2019-06-16_15-08-27a01btwsy/progress.csv',
         'DQN011/DQN_srv_2019-06-16_15-21-24udyj1n07/progress.csv',
         'DQN012/DQN_srv_2019-06-16_19-43-30tuphlj0i/progress.csv']

labels = ['mini batch = 2048',
          'mini batch = 1024',
          'mini batch = 256',
          'mini batch = 64']

# make multiple plots alongside in on figure
fig = plt.figure()

for i in range(0, 4):
    path = path_ray + files[i]
    data = pandas.read_csv(filepath_or_buffer=path, delimiter=',', header=1, usecols=[2, 13], names=['reward_mean', 'timesteps'])
    plt.plot(data['timesteps'], data['reward_mean'], label=labels[i])

plt.legend()
plt.show()