import os
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize


# =============================================================================
# Input arguments
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--project_dir',default='../project_directory', type=str)
parser.add_argument('--dnn',default='alexnet',type=str)
parser.add_argument('--test_dataset',default='Zhang_Wamsley',type=str)
parser.add_argument('--dream_idx',default=0, type=int)
parser.add_argument('--st',default='s', type=str)
args = parser.parse_args()

print(f'>>> Plot the RDMs of {args.test_dataset} <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
	print('{:16} {}'.format(key, val))
	
# =============================================================================
# Load correlation results
# =============================================================================

# Dream correlation scores list
ZW_corr_dir = os.path.join(args.project_dir, 'results', 'Zhang_Wamsley','REMs',
						  'correlation_scores_'+args.st)
dreams_corrs = os.listdir(ZW_corr_dir)

# Load correlation scores
RDMs = []
for c in dreams_corrs:
	result = np.load(os.path.join(ZW_corr_dir, c), allow_pickle=True).item()
	mean_corr = result['mean_correlations']
	RDMs.append(mean_corr)
	del result, mean_corr
RDMs = np.array(RDMs)
print(RDMs.shape)

# Dream images list
ZW_img_dir = os.path.join(args.project_dir, 'eeg_dataset', 'dream_data', 
						  'Zhang_Wamsley', 'images')
dreams_imgs = os.listdir(ZW_img_dir)
dreams_imgs = [s[6:].split('_')[0] for s in dreams_imgs]
	
# =============================================================================
# Plot the full RDMs
# =============================================================================

fig = plt.figure(figsize=(10, 4))
im = plt.imshow(RDMs, cmap='viridis',
				extent=[0, RDMs.shape[1], 0, RDMs.shape[0]], 
                origin='lower', aspect='auto')
cbar = plt.colorbar(im)
cbar.set_label('Values')

# Horizontal borders
for i in range(RDMs.shape[0]):
	plt.plot([0, RDMs.shape[1]], [i,i], 'k--', lw=0.4)

# Vertical borders
for i in range(RDMs.shape[0]):
	plt.plot([int(i*8),int(i*8)], [0, RDMs.shape[0]], 'k--', lw=0.4)

plt.xlabel('Images')
plt.ylabel('Dreams')
plt.title(f'REMs RDMs')
fig.tight_layout()
plt.show()

# =============================================================================
# Plot the max RDMs
# =============================================================================

# Get the maximum value among 8 images
num_imgs = 8
max_RDMs = np.empty((RDMs.shape[0], RDMs.shape[0]))
for v in range(RDMs.shape[0]):
    previous_cumu = 0
    current_cumu = 0
    for h in range(RDMs.shape[0]):
        current_cumu += num_imgs
        # print(f'({v},{h}): {previous_cumu}, {current_cumu}')
        max_RDMs[v,h] = max(RDMs[v, previous_cumu:current_cumu])
        previous_cumu += num_imgs
    del previous_cumu, current_cumu

# Normalization
norm_max_RDMs = normalize(max_RDMs)

# Plot
fig = plt.figure(figsize=(8, 8))
im = plt.imshow(max_RDMs, cmap='viridis',
				extent=[0, RDMs.shape[0], 0, RDMs.shape[0]], 
                origin='lower', aspect='auto')
cbar = plt.colorbar(im)
cbar.set_label('Values')

plt.xlim([0,RDMs.shape[0]])
plt.ylim([0,RDMs.shape[0]])
plt.xlabel('Images')
plt.ylabel('Dreams')
plt.title(f'max REMs RDMs')

fig.tight_layout()
plt.show()