import os
import argparse
import numpy as np

from img_EEG_func import concepts_select
from img_EEG_func import img_EEG_match

parser = argparse.ArgumentParser()
parser.add_argument('--project_dir', default='../project_directory', type=str)
parser.add_argument('--category', default='animal', type=str)
args = parser.parse_args()

print('>>> Matching THINGS higher level category with EEG data <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
	
#####################################################################
# Select concepts
#####################################################################
training_concepts = concepts_select(args, 'training')
test_concepts = concepts_select(args, 'test')
	
#####################################################################
# Match concepts with EEG 
#####################################################################
sub_list = list(range(1,11))

# Match the training data 
training_data = []
for s in sub_list:
    data = img_EEG_match(args, training_concepts, s, 'training')
    training_data.append(data)
    del data
training_data = np.array(training_data)
print(training_data.shape)

# Match the test data 
test_data = []
for s in sub_list:
    data = img_EEG_match(args, test_concepts, s, 'test')
    test_data.append(data)
    del data
test_data = np.array(test_data)
print(test_data.shape)