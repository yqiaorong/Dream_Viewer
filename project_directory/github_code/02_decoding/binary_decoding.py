import argparse
import numpy as np
from binary_decoding_func import feature_select
from binary_decoding_func import binary_classifier


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--project_dir', default='../project_directory', type=str)
parser.add_argument('--PSG_number', default='295', type=str)
parser.add_argument('--rem', default='1', type=str)
parser.add_argument('--obj_prop', default='pleasant_mean', type=str)
args = parser.parse_args()

print('>>> Binary classification <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
    
    
# =============================================================================
# Feature extraction
# =============================================================================
X_1, Y_1 = feature_select(args, 1)
X_2, Y_2 = feature_select(args, 2)
X = np.concatenate((X_1, X_2), axis=0)
Y = np.concatenate((Y_1, Y_2), axis=0)
                
# =============================================================================
# binary classification
# =============================================================================
binary_classifier(args, X, Y)