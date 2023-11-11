import os
import pandas as pd
num_slaves = int(os.environ['NUM_SLAVES'])
print(num_slaves)
#Load Dataset
cols = list(range(2,14))
df = pd.read_csv('spotify_songs.csv',usecols=cols)
#Partition the Dataset 
dir_path = 'Slave/Partitions'
if not os.path.exists(dir_path):
    # If it doesn't exist, create the directory

    os.makedirs(dir_path)
file_path = '/partition'
df_shuffled = df.sample(frac=1,random_state=23)
rows_per_part =  len(df_shuffled)//num_slaves
partitions = [df_shuffled.iloc[i:i+rows_per_part] for i in range(0,len(df_shuffled),rows_per_part)]
#Save the partitions
for i, partitions_df in enumerate(partitions,start=1):
    partition_filename = f'{dir_path+file_path}{i}.csv'
    partitions_df.to_csv(partition_filename,index=False)

print(f'Saved {num_slaves} partitions as CSV files!')