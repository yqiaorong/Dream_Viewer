import os

# Extract feature maps
os.system(f'python ../project_directory/github_code/03_dnn_fmaps/alexnet_ZW_REMs.py')
os.system(f'python ../project_directory/github_code/03_dnn_fmaps/pca.py --dataset ZW_REMs')

# Compute the correlation scores
os.system(f'python ../project_directory/github_code/04_validation_test/corr_ZW_REM_new.py')

# Plot the RDMs
os.system(f'python ../project_directory/github_code/05_plots/RDMs_REM_new.py')