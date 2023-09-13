import os
import argparse
import numpy as np
from tqdm import tqdm

from img_EEG_func import concepts_select, img_EEG_match
from img_EEG_func import load_dream_EEG, sort_THINGS, correlation_and_plot

parser = argparse.ArgumentParser()
parser.add_argument('--project_dir', default='../project_directory', type=str)
parser.add_argument('--category', default='food', type=str)
args = parser.parse_args()

print('>>> Matching THINGS higher level category with EEG data <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
	
##############################################################################
# Select concepts
##############################################################################

training_concepts = concepts_select(args, 'training')
test_concepts = concepts_select(args, 'test')
	
##############################################################################
# Match concepts with EEG 
##############################################################################

# The list of all THINGS subjects
sub_list = list(range(1,11))

d1, c1, d2, c2, THINGS_ch, THINGS_t = img_EEG_match(args, training_concepts, 
                                                    test_concepts, 1, subset=True)
# Capitalize THINGS channel names
THINGS_ch = [ch.upper() for ch in THINGS_ch]

# Create the arrays
train = np.empty((len(sub_list), *d1.shape))
train_compli = np.empty((len(sub_list), *c1.shape))
test = np.empty((len(sub_list), *d2.shape))
test_compli = np.empty((len(sub_list), *c2.shape))

# Iterate over all subjects
for ii, s in enumerate(tqdm(sub_list)):

    d1, c1, d2, c2, _, _ = img_EEG_match(args, training_concepts, 
                                         test_concepts, s, subset=True)
    train[ii] = d1
    train_compli[ii] = c1
    test[ii] = d2
    test_compli[ii] = c2

    del d1, c1, d2, c2

# Reshape all selected data
train = train.reshape(-2, *train.shape[3:])
train_compli = train_compli.reshape(-2, *train_compli.shape[3:])
test = test.reshape(-2, *test.shape[3:])
test_compli = test_compli.reshape(-2, *test_compli.shape[3:])

# Combine training and test data under the same category
category_data = np.concatenate((train, test), axis=0)
non_category_data = np.concatenate((train_compli, test_compli), axis=0)

# Drop extra THINGS channel and crop THINGS EEG data
sorted_THINGS_EEG_data, THINGS_t = sort_THINGS(category_data, THINGS_t)

##############################################################################
# Load dream EEG data
##############################################################################

# Input the category that you want to analyse
input_category = str(input(f"Please enter the category of dream that you want to analyse [{args.category}/NA]: "))
# Load the corresponding dream EEG data directory
dream_dir = os.path.join(args.project_dir, 'eeg_dataset', 'dream_data', 
                        'Zhang_Wamsley', 'preprocessed_data', input_category)
# The list of filenames in that directory
dream_subjs = os.listdir(dream_dir)
print(f'The number of dreams under the category {input_category}: {len(dream_subjs)}')

# Iterate over dreams under the same category
for subj in tqdm(dream_subjs):
    dream_EEG_data, dream_ch, dream_t = load_dream_EEG(dream_dir, subj)
    correlation_and_plot(args, subj, dream_t, dream_EEG_data, sorted_THINGS_EEG_data)