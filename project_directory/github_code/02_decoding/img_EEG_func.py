def concepts_select(args, img_type):
    import os
    import pandas as pd

    ### Load the image meta data with higher level categories ###
    THINGS_dir = os.path.join(args.project_dir, 'eeg_dataset', 'wake_data', 'THINGS')
    img_meta_dir = os.path.join(THINGS_dir, 'category53_longFormat.tsv')
    img_meta_data = pd.read_csv(img_meta_dir, delimiter='\t')
    
    ### First select concepts under one category ###
    # Notice: I choose 'Top-down Category (WordNet)' as reference for categorization
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

def img_EEG_match(args, concepts, sub, img_type):
    import os
    import numpy as np
    
    ### Load THINGS prepr data ###
    THINGS_prepr_dir = os.path.join(args.project_dir, 'eeg_dataset', 'wake_data', 'THINGS',
    'preprocessed_data', 'occipital', 'sub-'+format(sub,'02'),'preprocessed_eeg_'+img_type+'.npy')
    data = np.load(THINGS_prepr_dir, allow_pickle=True).item()
    data = data['preprocessed_eeg_data']

    ### Select data ###
    idx = [int(c[:5])-1 for c in concepts]

    if img_type == 'training':
        
        # Select data in the right category
        final_data = np.empty((len(idx)*10, *data.shape[1:]))
        for ii, i in enumerate(idx):
            final_data[ii*10:ii*10+10,:,:,:] = data[i*10:i*10+10,:,:,:]

        # Select data not in the right category
        idx_all = list(range(int(data.shape[0]/10)))
        idx_compli = [i for i in idx_all if i not in idx]
        compli_data = np.empty((len(idx_compli)*10, *data.shape[1:]))
        for ii, i in enumerate(idx_compli):
            compli_data[ii*10:ii*10+10,:,:,:] = data[i*10:i*10+10,:,:,:]

    elif img_type == 'test':

        # Select data in the right category
        final_data = np.empty((len(idx), *data.shape[1:]))
        for ii, i in enumerate(idx):
            final_data[ii,:,:,:] = data[i,:,:,:]

        # Select data not in the right category
        idx_compli = np.ones(data.shape[0], dtype=bool)
        idx_compli[idx] = False
        compli_data = data[idx_compli]

    return final_data, compli_data