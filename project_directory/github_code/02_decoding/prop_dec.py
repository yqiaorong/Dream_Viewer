import argparse
import numpy as np

from prop_dec_func import feature_select
from prop_dec_func import LearnCurve
from prop_dec_func import prop_dec

parser = argparse.ArgumentParser()
parser.add_argument('--project_dir', default='../project_directory', type=str)
parser.add_argument('--img_type', default='training', type=str)
parser.add_argument('--obj_prop', default='pleasant_mean', type=str)
parser.add_argument('--electrodes', default='occipital', type=str)
parser.add_argument('--rand_t_idx', default=40, type=int)
parser.add_argument('--n_perm', default=100, type=int)
args = parser.parse_args()

print('>>> THINGS EEG decoding <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
	
#####################################################################
# THINGS EEG training data
#####################################################################

X_1, Y_1, _, _ = feature_select(args,1,'training')
X_2, Y_2, _, _ = feature_select(args,2,'training')
X_3, Y_3, _, _ = feature_select(args,3,'training')

X = np.concatenate((X_1, X_2, X_3), axis=0)
Y = np.concatenate((Y_1, Y_2, Y_3), axis=0)

print('The shape of X_train:', X.shape)
print('The shape of y_train:',Y.shape)

# Shuffled X and Y
train_idx = np.arange(X.shape[0])  
np.random.shuffle(train_idx) 

X_shuffled = X[train_idx]
Y_shuffled = Y[train_idx]

# Plot the learning curve 
LearnCurve(X_shuffled, Y_shuffled, args)

#####################################################################
# THINGS EEG test data
#####################################################################

Xtest, Ytest, channels, times = feature_select(args,4,'test')
print('The shape od X_test:', Xtest.shape)
print('The shape od y_test:', Ytest.shape)

# Shuffled Xtest and Ytest
test_idx = np.arange(Xtest.shape[0])  
np.random.shuffle(test_idx) 

Xtest_shuffled = Xtest[test_idx]
Ytest_shuffled = Ytest[test_idx]

#####################################################################
# Decoding
#####################################################################
dec_result = prop_dec(X_shuffled, Y_shuffled, Xtest_shuffled, Ytest_shuffled, args, times)