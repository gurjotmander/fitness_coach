### This class adds labels the labels "squatting" and "standing" to the keypoint.csv and saves
### it as dataset.csv with the appropriate data separated into those labels.

import pandas as pd
import os

df = pd.read_csv('C:/Users/gurjo/Documents/term 8/major project/fitness app/Model/data/keypoints.csv')

dataset_path = 'C:/Users/gurjo/Documents/term 8/major project/fitness app/Model/data/images/'
squatting_path = os.path.join(dataset_path, 'squatting')
standing_path = os.path.join(dataset_path, 'standing')


# Functions for specifying labels based on file names
def get_label(image_name, squatting_path, standing_path):
    if image_name in os.listdir(squatting_path):
        return 'squatting'
    elif image_name in os.listdir(standing_path):
        return 'standing'
    else:
        return None

# Add label coloumn to our csv file
df['label'] = df['image_name'].apply(lambda x: get_label(x, squatting_path, standing_path))
df.to_csv(f'{dataset_path}dataset.csv', index=False)