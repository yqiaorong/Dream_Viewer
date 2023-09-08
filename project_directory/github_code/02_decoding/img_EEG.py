import argparse
from tqdm import tqdm
import numpy as np

from img_EEG_func import concepts_select
from img_EEG_func import img_EEG_match

parser = argparse.ArgumentParser()
parser.add_argument('--project_dir', default='../project_directory', type=str)
parser.add_argument('--category', default='food', type=str)
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

data_1, compli_1 = img_EEG_match(args, training_concepts, 1, 'training')
data_2, compli_2 = img_EEG_match(args, test_concepts, 1, 'test')

train = np.empty((len(sub_list), *data_1.shape))
train_compli = np.empty((len(sub_list), *compli_1.shape))
test = np.empty((len(sub_list), *data_2.shape))
test_compli = np.empty((len(sub_list), *compli_2.shape))

for ii, s in enumerate(tqdm(sub_list)):

    data_1, compli_1 = img_EEG_match(args, training_concepts, s, 'training')
    data_2, compli_2 = img_EEG_match(args, test_concepts, s, 'test')

    train[ii] = data_1
    train_compli[ii] = compli_1
    test[ii] = data_2
    test_compli[ii] = compli_2

    del data_1, compli_1, data_2, compli_2

# training_data = training_data.reshape(-2, *training_data.shape[3:])
# training_compli_data = training_compli_data.reshape(-2, *training_compli_data.shape[3:])
# test_data = test_data.reshape(-2, *test_data.shape[3:])
# test_compli_data = test_compli_data.reshape(-2, *test_compli_data.shape[3:])

print(f'The shape of all THINGS EEG training data under category {args.category}', train.shape, train_compli.shape)
print(f'The shape of all THINGS EEG test data under category {args.category}', test.shape, test_compli.shape)