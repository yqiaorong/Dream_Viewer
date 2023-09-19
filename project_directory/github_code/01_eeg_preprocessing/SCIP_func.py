import os
import sys
import argparse
from mne.io import read_raw_fif


# =============================================================================
# Input arguments
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--project_dir', default='../project_directory', type=str)
parser.add_argument('--subj', default='10', type=str)
parser.add_argument('--session', default='1', type=str)
parser.add_argument('--sfreq', default=100, type=int)
parser.add_argument('--mvnn_dim', default='time', type=str)
args = parser.parse_args()

print('>>> SCIP EEG data preprocessing <<<')
print('\nInput arguments:')
for key, val in vars(args).items():
    print('{:16} {}'.format(key, val))

ica_components = 50

# Parameters
cap_size = "L"
bad_chans = [] #can be entered manually or found automatically

SCIP_dir = os.path.join(args.project_dir, 'eeg_dataset', 'wake_data', 'SCIP', 'derivatives',
                        'preprocessed', 'sub-0'+args.subj, 'ses-0'+args.session, 'eeg')
eeg_dir = os.path.join(SCIP_dir, 'sub'+args.subj+'_sess'+args.session+'_50_ica_eeg.fif')
print(eeg_dir)

if os.path.exists(eeg_dir):
    raw = read_raw_fif(eeg_dir, preload=True) # creates two types, eeg and eog
    print(raw)
    print(raw.info)
    raw.pick_types(eog=False, eeg=True) # ignore eog channels
    print('Sample rate of the data is:', raw.info['sfreq'], 'Hz. It should be 1024 Hz')
else:
    print("There is not .set file for subject-session ", args.subj, args.session)
    sys.exit()