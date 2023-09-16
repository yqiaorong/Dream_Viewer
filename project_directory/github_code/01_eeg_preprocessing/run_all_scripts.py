import os

# =============================================================================
# THINGS1 EEG preprocessing
# =============================================================================
for s in range(1,51):
    os.system(f'python ../project_directory/github_code/01_eeg_preprocessing/THINGS1.py --subj {s}')

# =============================================================================
# THINGS2 EEG preprocessing
# =============================================================================
for s in range(1,11):
    os.system(f'python ../project_directory/github_code/01_eeg_preprocessing/THINGS2.py --subj {s}')