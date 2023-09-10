import os
import argparse
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.stats import pearsonr as corr

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

##############################################################################
# Sort THINGS and dream channels
##############################################################################

# Load dream data directory
dream_dir = os.path.join(args.project_dir, 'eeg_dataset', 'dream_data', 
                         'Zhang_Wamsley', 'preprocessed_data', args.category)
# The list of filenames in the directory
dream_subs = os.listdir(dream_dir)
print(f'The number of dreams under the category {args.category}: {len(dream_subs)}')
# Load the dream data
dream_sub = int(input("Please enter the idx of dream: "))
dream_data = np.load(os.path.join(dream_dir, dream_subs[dream_sub]), allow_pickle=True).item()
# Load the dream channel names
dream_ch = dream_data['ch_names']
# Load the dream times
dream_t = dream_data['times']
# Capitalize and shorten dream channel names
dream_ch = [ch[:-4].upper() for ch in dream_ch]
# Extract the index of the channel in THINGS which presents in THINGS but not in dream
redundant_ch = next(i for i, ch in enumerate(THINGS_ch) if ch not in dream_ch)
# Drop the redundant channel from THINGS channels 
THINGS_ch.pop(redundant_ch) 
# Sort the dream channel indices
dream_idx = [THINGS_ch.index(ch) for ch in dream_ch]
# Drop the THINGS EEG data of the redundant channels
category_data = np.delete(category_data, redundant_ch, axis=1)
non_category_data = np.delete(non_category_data, redundant_ch, axis=1)
# Drop the THINGS EEG times before t=0s
THINGS_t = THINGS_t[20:]
# Drop the THINGS EEG data before t=0s
category_data = category_data[:,:,20:]
non_category_data = non_category_data[:,:,20:]
# Load the dream EEG data 
dream_data = dream_data['preprocessed_eeg_data']
# Sort dream EEG data according to the dream channel indices
sorted_dream_data = np.empty((dream_data.shape))
for ii, i in enumerate(dream_idx):
    sorted_dream_data[ii] = dream_data[i] 
del dream_data

##############################################################################
# Correlations
##############################################################################

correlation = np.empty((len(THINGS_ch), len(dream_t)-len(THINGS_t)))
# Average the category data from THINGS
mean_category_data = np.mean(category_data, axis=0)
# Iterate over times 
for t in tqdm(range(len(dream_t)-len(THINGS_t))):
    # Iterate over channels
    for c in range(len(THINGS_ch)):
        correlation[c,t] = corr(mean_category_data[c,:], sorted_dream_data[c,t:t+len(THINGS_t)])[0]

# Plot the mean result
plt.figure()
plt.plot(dream_t[:-80], np.mean(correlation, axis=0))
plt.xlabel('Time (s)')
plt.ylabel('Pearson\'s $r$')
plt.ylim(bottom=-1, top=1)
plt.title('Correlation score')
plt.show()

# Plot the result
plt.figure(figsize=(8, 6))
plt.imshow(correlation, cmap='viridis', extent=[0, len(dream_t)-len(THINGS_t), 0, len(dream_ch)], origin='lower', aspect='auto')
cbar = plt.colorbar()
cbar.set_label('Values')
plt.xlabel('Time(s)')
plt.ylabel('Channels')
plt.title('Correlation score')
plt.show()

# Plot the dream EEG 
plt.figure()
for c in range(len(dream_ch)):
    plt.plot(dream_t[:80], sorted_dream_data[c,:80], alpha=0.2)
plt.xlabel('Time (s)')
plt.title('dream')
plt.show()