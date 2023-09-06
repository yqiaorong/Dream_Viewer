import os
from tqdm import tqdm

project_dir = os.path.join('D:/', 'UG', 'Research', 'Dream_Neural_Decoding',
                           'Shannon_Big_Big_Dream_Project','project_directory')

# =============================================================================
# DEED
# =============================================================================
DEED_dir = os.path.join('eeg_dataset', 'dream_data', 'DEED', 'raw_data', 'raw_mat') 

filenames = os.listdir(os.path.join(project_dir, DEED_dir))
print('')
print('=====================================================================')
print('>>> DEED preprocessing <<<')
print('=====================================================================')
print('')
for f in tqdm(filenames, desc='Preprocessing DEED raw files'):
    os.system(f'python github_code/01_eeg_preprocessing/DEED.py --PSG_number {f}')