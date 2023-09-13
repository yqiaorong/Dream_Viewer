def concepts_select(args, img_type):
    import os
    import pandas as pd

    ### Load the image meta data with higher level categories ###
    THINGS_dir = os.path.join(args.project_dir, 'eeg_dataset', 'wake_data', 'THINGS')
    img_meta_dir = os.path.join(THINGS_dir, 'category53_longFormat.tsv')
    img_meta_data = pd.read_csv(img_meta_dir, delimiter='\t')
    
    ### First select concepts under one category ###
    df = img_meta_data[img_meta_data['category'] == args.category]
    init_select = []
    init_select.extend(df['uniqueID'].tolist())
    print(f'The total number of concepts under category {args.category}: ', len(init_select))

    ### Secondly select concepts only appeared in training/test set ###
    img_dir  = os.path.join(THINGS_dir, 'image_set', img_type+'_images') 
    concepts = os.listdir(img_dir)
    
    final_select = []
    for i_s in init_select:
        for c in concepts:
            if i_s == c[6:]:
                final_select.append(c)
    print(f'The number of concepts under category {args.category} in {img_type} set: ', len(final_select))

    return final_select

def img_EEG_match(args, train_concepts, test_concepts, sub, subset=False):
    import os
    import random
    import numpy as np
    
    THINGS_dir = os.path.join(args.project_dir, 'eeg_dataset', 'wake_data', 'THINGS',
    'preprocessed_data', 'occipital', 'sub-'+format(sub,'02'))

    ### Load THINGS prepr training data ###
    THINGS_train_dir = os.path.join(THINGS_dir, 'preprocessed_eeg_training.npy')
    train_data = np.load(THINGS_train_dir, allow_pickle=True).item()
    # Load channels and times
    channels = train_data['ch_names']
    times = train_data['times']
    train_data = train_data['preprocessed_eeg_data']
    
    ### Select training data ###
    # The training idx under the category
    train_idx = [int(c[:5])-1 for c in train_concepts]

    # Select data in the right category
    train = np.empty((len(train_idx)*10, *train_data.shape[1:]))
    for ii, i in enumerate(train_idx):
        train[ii*10:ii*10+10,:,:,:] = train_data[i*10:i*10+10,:,:,:]
    
    # Select data not in the right category
    train_idx_all = list(range(int(train_data.shape[0]/10)))
    train_idx_compli = [i for i in train_idx_all if i not in train_idx]
    if subset == True:
        train_idx_compli = random.sample(train_idx_compli, len(train_idx))
    else:
        pass
    train_compli = np.empty((len(train_idx_compli)*10, *train_data.shape[1:]))
    for ii, i in enumerate(train_idx_compli):
        train_compli[ii*10:ii*10+10,:,:,:] = train_data[i*10:i*10+10,:,:,:]

    ### Load THINGS prepr test data ###
    THINGS_test_dir = os.path.join(THINGS_dir, 'preprocessed_eeg_test.npy')
    test_data = np.load(THINGS_test_dir, allow_pickle=True).item()
    test_data = test_data['preprocessed_eeg_data']

    ### Select test data ###
    # The test idx under the category
    test_idx = [int(c[:5])-1 for c in test_concepts]

    # Select data in the right category
    test = np.empty((len(test_idx), *test_data.shape[1:]))
    for ii, i in enumerate(test_idx):
        test[ii,:,:,:] = test_data[i,:,:,:]

    # Select data not in the right category
    test_idx_all = list(range(int(test_data.shape[0])))
    test_idx_compli = [i for i in test_idx_all if i not in test_idx]
    if subset == True:
        test_idx_compli = random.sample(test_idx_compli, len(test_idx))
    else:
        pass
    test_compli = np.empty((len(test_idx_compli), *test_data.shape[1:]))
    for ii, i in enumerate(test_idx_compli):
        test_compli[ii,:,:,:] = test_data[i,:,:,:]

    return train, train_compli, test, test_compli, channels, times

def load_dream_EEG(dream_dir, dream_sub):
    """
    Parameters
    ----------
    dream_dir ; str
        The directory of preprocessed EEG data with the same category of dream dataset.
    dream_sub : int
        The idx of target dream under the category that you select for analysis.

    Returns
    -------
    dream_EEG_data: array with shape (channels, times)
        The dream EEG data.
    dream_ch : list of str
        The EEG channels in dream dataset.
    dream_t : list of float
        The time points in dream dataset.
    """

    import os
    import numpy as np

    # Load the dream data
    dream_data = np.load(os.path.join(dream_dir, dream_sub), allow_pickle=True).item()
    # Load the dream channel names
    dream_ch = dream_data['ch_names']
    # Load the dream times
    dream_t = dream_data['times']
    # Load the dream EEG data 
    dream_EEG_data = dream_data['preprocessed_eeg_data']

    del dream_data
    return dream_EEG_data, dream_ch, dream_t

def sort_THINGS(THINGS_EEG_data, THINGS_t):
    """
    Parameters
    ----------
    THINGS_EEG_data: array with shape (images, channels, times)
        The THINGS EEG data
    THINGS_t: list of float
        The original THINGS EEG times

    Returns
    -------
    sorted_THINGS_EEG_data: array with shape (images, channels, times)
        The sorted THINGS EEG data
        --redundant channel is droped
        --croped EEG data
    THINGS_t : list of float
        The croped THINGS EEG times
    """
    import numpy as np

    # Drop the THINGS EEG data of the redundant channels
    redundant_ch = 12
    sorted_THINGS_EEG_data = np.delete(THINGS_EEG_data, redundant_ch, axis=1)
    # Drop the THINGS EEG times before t=0s
    THINGS_t = THINGS_t[20:]
    # Drop the THINGS EEG data before t=0s
    sorted_THINGS_EEG_data = sorted_THINGS_EEG_data[:,:,20:]

    del THINGS_EEG_data
    return sorted_THINGS_EEG_data, THINGS_t


def correlation_and_plot(args, dream_subj, dream_t, sorted_dream_EEG_data, sorted_THINGS_EEG_data):
    """
    Parameters
    ----------
    args : Namespace
        Input arguments.
    dream_subj: int
        The idx of target dream under the category that you select for analysis.
    dream_t: list of float
        The time points in dream dataset.
    sorted_dream_EEG_data: array with shape (channels, times)
        The dream EEG data
    sorted_THINGS_EEG_data: array with shape (images, channels, times)
        The THINGS EEG data

    Returns
    -------
    
    """
    import os
    import numpy as np
    from matplotlib import pyplot as plt
    from scipy.stats import pearsonr as corr
    
    ### Correlations ###
    # The correlation result array
    correlation = np.empty((len(dream_t)))
    # Average the category data from THINGS EEG across concepts 
    mean_THINGS_data = np.mean(sorted_THINGS_EEG_data, axis=0)
    # Average the category data from THINGS EEG across time 
    mean_THINGS_data = np.mean(mean_THINGS_data[:,10:30], axis=1)
    # Iterate over times 
    for t in range(len(dream_t)):
        correlation[t] = corr(sorted_dream_EEG_data[:,t], mean_THINGS_data)[0]
    ztrans_corr = np.arctanh(correlation)
    
    ### Plot the correlation results ###
    # Create a figure 
    figure_size = (12, 6)
    fig = plt.figure(figsize=figure_size)
    # Define the grid layout using GridSpec
    gs = plt.GridSpec(2, 2, figure=fig, width_ratios=[3, 1])

    # Subplot 1: Temporal correlation scores for the first 20 time points
    ax1 = plt.subplot(gs[0, 0])
    ax1.plot(dream_t[:2000], ztrans_corr[:2000], color='skyblue')
    ax1.set_title('Temporal Correlation')
    ax1.set_ylabel('Correlation Score')

    # Subplot 2: Histogram of correlation scores for the first 20 time points
    ax2 = plt.subplot(gs[0, 1])
    ax2.hist(ztrans_corr[:2000], bins=10, color='skyblue', edgecolor='black')
    ax2.set_title('Histogram of Correlation Scores')
    ax2.set_ylabel('Frequency')

    # Subplot 3: Temporal correlation scores for the last 20 time points
    ax3 = plt.subplot(gs[1, 0])
    ax3.plot(dream_t[-2000:], ztrans_corr[-2000:], color='salmon')
    ax3.set_xlabel('Time Points')
    ax3.set_ylabel('Correlation Score')

    # Subplot 4: Histogram of correlation scores for the last 20 time points
    ax4 = plt.subplot(gs[1, 1])
    ax4.hist(ztrans_corr[-2000:], bins=10, color='salmon', edgecolor='black')
    ax4.set_xlabel('Correlation Score')
    ax4.set_ylabel('Frequency')

    plt.tight_layout()
    
    # Save figure
    save_folder = os.path.join(args.project_dir, 'results', args.category)
    if os.path.isdir(save_folder) == False:
        os.makedirs(save_folder)
    save_dir = os.path.join(save_folder, dream_subj[6:-4])
    plt.savefig(save_dir)