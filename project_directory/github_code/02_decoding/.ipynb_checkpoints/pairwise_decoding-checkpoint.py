"""Pairwise decoding of the synthetic EEG test data, and noise ceiling
calculation. For each EEG time point, a SVM classifier is trained to decode
between each combination of two biological data image conditions (using the EEG
channels data), and is then tested on the corresponding combinations of two
synthetic data image conditions.

Parameters
----------
sub : int
	Used subject.
encoding_type : str
	Whether to analyze the 'linearizing' or 'end-to-end' encoding synthetic
	data.
dnn : str
	Used DNN network.
pretrained : bool
	If True, analyze the data synthesized through pretrained (linearizing or
	end-to-end) models. If False, analyze the data synthesized through randomly
	initialized (linearizing or end-to-end) models.
subjects : str
	If 'linearizing' encoding_type is chosen, whether to analyze the 'within' or
	'between' subjects linearizing encoding synthetic data.
layers : str
	If 'linearizing' encoding_type is chosen, whether to analyse the data
	synthesized using 'all', 'single' or 'appended' DNN layers feature maps.
n_components : int
	If 'linearizing' encoding_type is chosen, number of DNN feature maps PCA
	components retained for synthesizing the EEG data.
modeled_time_points : str
	If 'end_to_end' encoding_type is chosen, whether to analyze the synthetic
	data of end-to-end models trained to predict 'single' or 'all' time points.
lr : float
	If 'end_to_end' encoding_type is chosen, learning rate used to train the
	end-to-end encoding models.
weight_decay : float
	If 'end_to_end' encoding_type is chosen, weight decay coefficint used to
	train the end-to-end encoding models.
batch_size : int
	If 'end_to_end' encoding_type is chosen, batch size used to train the
	end-to-end encoding models.
n_iter : int
	Number of analysis iterations.
project_dir : str
	Directory of the project folder.

"""

import argparse
import os
import numpy as np
from tqdm import tqdm
from sklearn.utils import resample
from sklearn.svm import SVC


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--sub', default=1, type=int)

parser.add_argument('--subjects', default='within', type=str)

parser.add_argument('--modeled_time_points', type=str, default='single')
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--weight_decay', type=float, default=0.)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_iter', default=100, type=int)
parser.add_argument('--project_dir', default='../project_directory', type=str)
args = parser.parse_args()

print('>>> Pairwise decoding <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))

# Set random seed for reproducible results
seed = 20200220
np.random.seed(seed)


# =============================================================================
# Load the biological EEG test data
# =============================================================================
data_dir = os.path.join('eeg_dataset','wake_data','THINGS','preprocessed_data', 'sub-'+
                        format(args.sub,'02'), 'preprocessed_eeg_test.npy')

bio_data = np.load(os.path.join(args.project_dir, data_dir), allow_pickle=True).item()

bio_test = bio_data['preprocessed_eeg_data']
times = bio_data['times']
ch_names = bio_data['ch_names']

del bio_data


# # =============================================================================
# # Load the synthetic EEG test data
# # =============================================================================
data_dir = os.path.join('eeg_dataset','wake_data','THINGS','preprocessed_data', 'sub-'+
                        format(args.sub+1,'02'), 'preprocessed_eeg_test.npy')

synt_data = np.load(os.path.join(args.project_dir, data_dir), allow_pickle=True).item()

synt_test = synt_data['preprocessed_eeg_data']

del synt_data


# =============================================================================
# Compute the pairwise decoding and noise ceiling
# =============================================================================
# Results and noise ceiling matrices of shape:
# (Iterations × Image conditions × Image conditions × EEG time points)
#
# 1. pair_dec
# 2. noise_ceiling_low
# 3. noise_ceiling_up
#

img_cond = bio_test.shape[0]
rep = bio_test.shape[1]
channels = bio_test.shape[2]
time = bio_test.shape[3]

pair_dec  = np.zeros((args.n_iter, img_cond, img_cond, time))
noise_ceiling_low = np.zeros((args.n_iter, img_cond, img_cond, time))
noise_ceiling_up = np.zeros((args.n_iter, img_cond, img_cond, time))

# Average across all the biological data repetitions for the noise ceiling
# upper bound calculation
bio_data_avg_all = np.mean(bio_test, 1)

# Loop over iterations
for i in tqdm(range(args.n_iter)):
    # Random data repetitions index
    shuffle_idx = resample(np.arange(0, rep), replace=False, n_samples=int(rep/2)) # See notebook!!!



    # Select one half (40) of the biological data repetitions for training the
    # classifier, and average them into 10 pseudo-trials of 4 repetitions
    #
    # This part excludes shuffle_idx!!!
    #
    ptrial_rep = 4
    ptrail_num = int((rep/2) / ptrial_rep) # = 10 

    bio_data_avg_half_1 = np.zeros((img_cond, ptrail_num, channels, time))
    # shape (200, 10, 6, 100)

    bio_data_provv = np.delete(bio_test, shuffle_idx, 1) # See notebook!!!

    for r in range(ptrail_num): # r = 0,1,...,8,9
        bio_data_avg_half_1[:,r] = np.mean(
            bio_data_provv[:,r*ptrial_rep:r*ptrial_rep+ptrial_rep], 1)
    del bio_data_provv
    # This tells, we average the repetition 
    # from idx_0 to idx_3,---> bio_data_avg_half_1[:,0,:,:] 
    # from idx_4 to idx_7,---> bio_data_avg_half_1[:,1,:,:] 
    # ...
    # from idx_36 to idx_39,---> bio_data_avg_half_1[:,9,:,:] 



    # Average across the other half of the biological data repetitions for the
    # noise ceiling lower bound calculation
    #
    # This part includes shuffle_idx!!!
    #
    bio_data_avg_half_2 = np.mean(bio_test[:,shuffle_idx], 1) # averaged across 40 repetitions
    # shape (200, 1, 17, 100)
    
    # X_test will be the synthesed EEG data (averaged over repetitions
    synt_test = np.mean(synt_test, axis=1)
    

    ### Classifier target vectors ###
    y_train = np.zeros((ptrail_num)*2) # = 20
    y_train[ptrail_num:(ptrail_num)*2] = 1 
    # The first 10 elements of y_train = 0
    # The last 10 elements of y_train = 1

    y_test = np.asarray((0, 1))

    ### Loop over image-conditions and EEG time points ###
    for i1 in range(img_cond):
        for i2 in range(img_cond):
            if i1 < i2:
                for t in range(time):
                    # Train the classifier
                    X_train = np.append(bio_data_avg_half_1[i1,:,:,t],
                        bio_data_avg_half_1[i2,:,:,t], 0) # X_train (1,20,17,1)
                    dec_svm = SVC(kernel="linear")
                    dec_svm.fit(X_train, y_train) # y_train (20,): [0,...0,1,...1]

                    # Test the classifier on the synthetic data
                    X_test_synt = np.append(
                        np.expand_dims(synt_test[i1,:,t], 0), 
                        np.expand_dims(synt_test[i2,:,t], 0), 0)
                    # X_test_synt appends two arrays with shape (1,channels)

                    y_pred = dec_svm.predict(X_test_synt)
                    pair_dec[i,i2,i1,t] = sum(y_pred == y_test) / len(y_test)
                    # This is the pairwise decoding accuracy

                    # Compute the noise ceiling
                    X_test_avg_half = np.append(np.expand_dims(
                        bio_data_avg_half_2[i1,:,t], 0), np.expand_dims(
                        bio_data_avg_half_2[i2,:,t], 0), 0)
                    X_test_avg_all = np.append(np.expand_dims(
                        bio_data_avg_all[i1,:,t], 0), np.expand_dims(
                        bio_data_avg_all[i2,:,t], 0), 0)

                    y_pred_noise_ceiling_low = dec_svm.predict(X_test_avg_half)
                    y_pred_noise_ceiling_up = dec_svm.predict(X_test_avg_all)

                    noise_ceiling_low[i,i2,i1,t] = sum(
                        y_pred_noise_ceiling_low == y_test) / len(y_test)
                    noise_ceiling_up[i,i2,i1,t] = sum(
                        y_pred_noise_ceiling_up == y_test) / len(y_test)


# =============================================================================
# Average the results across iterations and pairwise comparisons
# =============================================================================
# Average across iterations
pair_dec = np.mean(pair_dec, 0)
noise_ceiling_low = np.mean(noise_ceiling_low, 0)
noise_ceiling_up = np.mean(noise_ceiling_up, 0)

# Average across pairwise comparisons
idx = np.tril_indices(img_cond, -1) # see notebook!!!

pair_dec = np.mean(pair_dec[idx], 0) # see notebook!!!
# The final shape of pair_dec[layer] is :
#
# (the number element in lower triangle of (img_cond,img_cond), time)
# which should be (199000,100)
#
# img_cond = 200
#
noise_ceiling_low = np.mean(noise_ceiling_low[idx], 0)
noise_ceiling_up = np.mean(noise_ceiling_up[idx], 0)


# =============================================================================
# Save the results
# =============================================================================
# Store the results into a dictionary
results_dict = {
    'pairwise_decoding': pair_dec,
    'noise_ceiling_low': noise_ceiling_low,
    'noise_ceiling_up': noise_ceiling_up,
    'times': times,
    'ch_names': ch_names
}

# Saving directory
save_dir = os.path.join(args.project_dir, 'results', 'sub-'+format(args.sub,'02'), 'pairwise_decoding')
file_name = 'pairwise_decoding.npy'

# Create the directory if not existing and save
if os.path.isdir(save_dir) == False:
    os.makedirs(save_dir)
np.save(os.path.join(save_dir, file_name), results_dict)
