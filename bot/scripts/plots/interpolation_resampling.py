
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

fig = plt.figure()

# process data 1
data1 = {'A': [0.99, 1.01, 1.99, 2.01], 'B': [0.3, 0.4, 0.7, 0.8]}

df = pd.DataFrame(data1)

n = 4
new_sample_points = np.linspace(0, n, 2*n+1)
df_new_sample_points = pd.DataFrame(new_sample_points, columns=list('A'))
df1 = df.append(df_new_sample_points, ignore_index=True, sort=True)
#df2 = df1.set_index('A')


df2 = df1.sort_values(by='A')
df3 = df2.interpolate(method='linear', axis=0)
print(df3)
df_erster_satz = df3.set_index('A')
plt.plot(df_erster_satz, label='erster DS')



# process data 2
data2 = {'A': [0.98, 1.0123, 1.69, 2.21, 2.9], 'B': [0.2, 0.3, 0.4, 0.6, 1.4]}

df = pd.DataFrame(data2)

n = 3
new_sample_points = np.linspace(0, n, 2*n+1)
df_new_sample_points = pd.DataFrame(new_sample_points, columns=list('A'))
df1 = df.append(df_new_sample_points, ignore_index=True, sort=True)
#df2 = df1.set_index('A')


df2 = df1.sort_values(by='A')
df3 = df2.interpolate(method='linear', axis=0)
print(df3)
df_zweiter_satz = df3.set_index('A')
plt.plot(df_zweiter_satz, label='zweiter DS')





# combine data to one sheet
df_alle = pd.DataFrame()
#experimental ################
df_alle['B1'] = df_erster_satz['B']
# df_alle = df_erster_satz
############### ende ############
df_alle['B2'] = df_zweiter_satz['B']

# lösche alle Samples, die nicht jeder Datensatz hat
# Nebeneffekt: beim Plot geht der mean auch nur bis zur Länge des kürzesten DS (ist aber nich schlimm)
df_alle = df_alle[np.isfinite(df_alle['B2'])]

print(df_alle)

print("MEAN")
df_mean = df_alle.mean(axis=1)
print(df_mean)
plt.plot(df_mean, label='mean shit')


plt.legend()
plt.show()
