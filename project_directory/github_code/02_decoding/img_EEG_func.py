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