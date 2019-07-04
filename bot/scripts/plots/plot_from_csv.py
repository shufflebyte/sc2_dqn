# Plot results from csv file

import matplotlib
# add this to use TKagg backend on MacOSX ..
# the fucking fuck.... jesus christ...
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import pandas
path_ray_results = '/Users/florian/ray_results/'
file = 'DQN003/DQN_srv_2019-06-14_18-02-00ok73e6of/progress.csv'
path = path_ray_results+file

# start counting in csv at 0
rewards_mean = pandas.read_csv(filepath_or_buffer=path, delimiter=',', header=1, usecols=[2], names=['reward_mean'])
rewards_max = pandas.read_csv(filepath_or_buffer=path, delimiter=',', header=1, usecols=[0], names=['reward_max'])
iterations = pandas.read_csv(filepath_or_buffer=path, delimiter=',', header=1, usecols=[15], names=['training_iterations'])
timesteps = pandas.read_csv(filepath_or_buffer=path, delimiter=',', header=1, usecols=[13], names=['timesteps'])
seconds = pandas.read_csv(filepath_or_buffer=path, delimiter=',', header=1, usecols=[20], names=['time_s'])

print(type(rewards_max))

#debug data
print(rewards_max)
# print(iterations)
# print(timesteps)
# print(seconds)

# scaling for the plot
minutes = seconds.div(60)
hours = minutes.div(60)

# make multiple lines in one plot
fig = plt.figure()
plt.plot(timesteps, rewards_mean, label='Reward over timesteps')
plt.plot(timesteps, iterations, label='iterations over timesteps (nonsense :-p)')
plt.legend()
plt.show()

# make multiple plots beside in on figure
fig2 = plt.figure()
fig2.suptitle("Comparison of the x-axis possiblities")

plt.subplot(1,3,1)
plt.plot(timesteps, rewards_mean)
plt.xlabel("timesteps")
plt.ylabel("reward")

plt.subplot(1,3,2)
plt.plot(iterations, rewards_mean)
plt.xlabel("training iterations")
plt.ylabel("reward")

plt.subplot(1,3,3)
plt.plot(hours, rewards_mean)
plt.xlabel("time in hours")
plt.ylabel("reward")

plt.show()
