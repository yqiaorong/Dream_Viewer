"""Grouping DEED dream raw files according to experiment participants."""

import os
import shutil


source_dir = os.path.join('D:/', 'UG', 'Research', 'Dream_Neural_Decoding',
                           'Shannon_Big_Big_Dream_Project', 'project_directory',
                           'eeg_dataset', 'dream_data', 'DEED', 'dream_raw_data')

for file in os.listdir(source_dir):

    # Extract the participant number from the file name
    participant_number = file[4:7]  

    # Create the subfolder if it doesn't exist
    save_dir = os.path.join(source_dir, participant_number)
    if os.path.isdir(save_dir) == False:
        os.makedirs(save_dir)

    # Move the file to the corresponding subfolder
    source_file_path = os.path.join(source_dir, file)
    save_file_path = os.path.join(save_dir, file)
    shutil.move(source_file_path, save_file_path)

    print(participant_number, "done")