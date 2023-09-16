import os
import argparse
import numpy as np
from tqdm import tqdm

from img_EEG_func import THINGS_concepts_select, img_EEG_match
from img_EEG_func import load_dream_EEG, sort_THINGS, correlation_and_plot

parser = argparse.ArgumentParser()
parser.add_argument('--project_dir', default='../project_directory', type=str)
args = parser.parse_args()

print('>>> Matching THINGS higher level category with EEG data <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
	
##############################################################################
# Select THINGS concepts
##############################################################################

# Input the THINGS category that you want to analyse
THINGS_cate = input('Please enter the THINGS category that you want to analyse: ')

training_concepts = THINGS_concepts_select(args, THINGS_cate, 'training')
test_concepts = THINGS_concepts_select(args, THINGS_cate, 'test')
	
##############################################################################
# Match THINGS concepts with THINGS EEG data
##############################################################################

# The list of all THINGS subjects
sub_list = list(range(1,11))

d1, _, d2, _, THINGS_ch, THINGS_t = img_EEG_match(args, training_concepts, 
                                                    test_concepts, 1, subset=True)
# Capitalize THINGS channel names
THINGS_ch = [ch.upper() for ch in THINGS_ch]

# Create the arrays
train = np.empty((len(sub_list), *d1.shape))
test = np.empty((len(sub_list), *d2.shape))

# Iterate over all subjects
for ii, s in enumerate(tqdm(sub_list)):

    d1, _, d2, _, _, _ = img_EEG_match(args, training_concepts, 
                                         test_concepts, s, subset=True)
    train[ii] = d1
    test[ii] = d2

    del d1, d2

# Reshape all selected data
train = train.reshape(-2, *train.shape[3:])
test = test.reshape(-2, *test.shape[3:])

# Combine training and test data under the same category
category_data = np.concatenate((train, test), axis=0)

# Drop extra THINGS channel and crop THINGS EEG data
sorted_THINGS_EEG_data, THINGS_t = sort_THINGS(category_data, THINGS_t)

##############################################################################
# Load dream EEG data
##############################################################################

# Input the category that you want to analyse
dream_cate = str(input(f"Please enter the category of dream that you want to analyse: "))
# Load the corresponding dream EEG data directory
dream_dir = os.path.join(args.project_dir, 'eeg_dataset', 'dream_data', 
                        'Zhang_Wamsley', 'preprocessed_data', dream_cate)
# The list of filenames in that directory
dream_subjs = os.listdir(dream_dir)
print(f'The number of dreams under the category {dream_cate}: {len(dream_subjs)}')

# Iterate over dreams under the same category
for subj in tqdm(dream_subjs):
    dream_EEG_data, dream_ch, dream_t = load_dream_EEG(dream_dir, subj)
    correlation_and_plot(args, dream_cate, THINGS_cate, subj, dream_t, dream_EEG_data, sorted_THINGS_EEG_data)                           