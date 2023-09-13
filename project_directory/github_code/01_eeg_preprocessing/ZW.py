import argparse
import mne
import os 

from ZW_get_categories_func import epoching, mvnn, save_prepr

parser = argparse.ArgumentParser()
parser.add_argument('--project_dir', default='../project_directory', type=str)
parser.add_argument('--PSG', default='010_Morning', type=str)
parser.add_argument('--category', default='NA', type=str)
parser.add_argument('--sfreq', default=100, type=int)
parser.add_argument('--chunk_size', default='100', type=int)
parser.add_argument('--mvnn_dim', default='time', type=str)
args = parser.parse_args()

##############################################################################
# Preprocess the raw EEG file 
##############################################################################

# The path of target PSG file
Zhang_dir = os.path.join(args.project_dir, 'eeg_dataset', 'dream_data',
                         'Zhang_Wamsley')
import_path_PSG = os.path.join(Zhang_dir, 'Data', 'PSG')

# Read the target PSG file
raw = mne.io.read_raw_edf(os.path.join(import_path_PSG, 'subject'+args.PSG+'.edf'))
data, ch_names, times = epoching(args, raw)
whitened_data = mvnn(args, data)
save_prepr(args, whitened_data, ch_names, times, args.PSG)