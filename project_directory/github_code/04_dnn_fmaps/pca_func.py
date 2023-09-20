def train_scaler_pca(args):
    """The function train the standardization and pca model on THINGS 
    training images.
    
    Parameters
    ----------
    args : Namespace
        Input arguments.

    Returns
    -------
    scaler : list of models
        the standardization model.
    pca : list of models
        the pca model.
    fmaps_train : 
        training image feature maps.
    all_layers : list
        The list of names of layers.
    layer_names : list
        The list of names of layers.
    """
        
    import os
    import numpy as np
    from tqdm import tqdm
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import KernelPCA

    # Set random seed for reproducible results
    seed = 20200220

    # Load the feature maps
    feats = []
    feats_all = []
    fmaps_train = {}
    fmaps_dir = os.path.join(args.project_dir, 'eeg_dataset','wake_data','THINGS_EEG2',
                            'dnn_feature_maps','full_feature_maps', args.dnn, 
                            'pretrained-'+str(args.pretrained),'training_images')
    fmaps_list = os.listdir(fmaps_dir)
    fmaps_list.sort()
    for f, fmaps in enumerate(tqdm(fmaps_list, desc='training_images')):
        fmaps_data = np.load(os.path.join(fmaps_dir, fmaps),
            allow_pickle=True).item()
        all_layers = fmaps_data.keys()
        if args.layers == 'all':
            layer_names = ['all_layers']
        elif args.layers == 'single':
            layer_names = all_layers
        for l, dnn_layer in enumerate(all_layers):
            if args.layers == 'all':
                if l == 0:
                    feats = np.reshape(fmaps_data[dnn_layer], -1)
                else:
                    feats = np.append(feats, np.reshape(fmaps_data[dnn_layer], -1))
            elif args.layers == 'single':
                if f == 0:
                    feats.append([[np.reshape(fmaps_data[dnn_layer], -1)]])
                else:
                    feats[l].append([np.reshape(fmaps_data[dnn_layer], -1)])
        if args.layers == 'all':
            feats_all.append(feats)
    if args.layers == 'all':
        fmaps_train[layer_names[0]] = np.asarray(feats_all)
    elif args.layers == 'single':
        for l, dnn_layer in enumerate(layer_names):
            fmaps_train[dnn_layer] = np.squeeze(np.asarray(feats[l]))

    # Standardize the data
    scaler = []
    for l, dnn_layer in enumerate(layer_names):
        scaler.append(StandardScaler())
        scaler[l].fit(fmaps_train[dnn_layer])

    # Apply PCA
    pca = []
    for l, dnn_layer in enumerate(layer_names):
        pca.append(KernelPCA(n_components=args.n_components, kernel='poly',
            degree=4, random_state=seed))
        pca[l].fit(fmaps_train[dnn_layer])

    return scaler, pca, fmaps_train, all_layers, layer_names

def apply_scaler_pca(args, img_category, scaler, pca, all_layers, layer_names):
    """The function apply the standardization and pca models on the target 
    images dataset.
    
    Parameters
    ----------
    args : Namespace
        Input arguments.
    img_category : str
        The image category within the used dataset.
    scaler : list of models
        the standardization model.
    pca : list of models
        the pca model.
    fmaps_train : 
        training image feature maps.
    all_layers : list
        The list of names of layers.
    layer_names : list
        The list of names of layers.
    """
        
    import os
    import numpy as np
    from tqdm import tqdm

    # Load the feature maps
    feats = []
    feats_all = []
    fmaps_test = {}
    fmaps_dir = os.path.join(args.project_dir,'eeg_dataset','wake_data',args.dataset,
                             'dnn_feature_maps','full_feature_maps',args.dnn,
                             'pretrained-'+str(args.pretrained),img_category)
    fmaps_list = os.listdir(fmaps_dir)
    fmaps_list.sort()
    for f, fmaps in enumerate(tqdm(fmaps_list, desc=img_category)):
        fmaps_data = np.load(os.path.join(fmaps_dir, fmaps), allow_pickle=True).item()
        for l, dnn_layer in enumerate(all_layers):
            if args.layers == 'all':
                if l == 0:
                    feats = np.reshape(fmaps_data[dnn_layer], -1)
                else:
                    feats = np.append(feats, np.reshape(fmaps_data[dnn_layer], -1))
            elif args.layers == 'single':
                if f == 0:
                    feats.append([[np.reshape(fmaps_data[dnn_layer], -1)]])
                else:
                    feats[l].append([np.reshape(fmaps_data[dnn_layer], -1)])
        if args.layers == 'all':
            feats_all.append(feats)
    if args.layers == 'all':
        fmaps_test[layer_names[0]] = np.asarray(feats_all)
    elif args.layers == 'single':
        for l, dnn_layer in enumerate(layer_names):
            fmaps_test[dnn_layer] = np.squeeze(np.asarray(feats[l]))

    # Standardize the data
    for l, dnn_layer in enumerate(layer_names):
        fmaps_test[dnn_layer] = scaler[l].transform(fmaps_test[dnn_layer])

    # Apply PCA
    for l, dnn_layer in enumerate(layer_names):
        fmaps_test[dnn_layer] = pca[l].transform(fmaps_test[dnn_layer])

    # Save the downsampled feature maps
    save_dir = os.path.join(args.project_dir,'eeg_dataset','wake_data',args.dataset,
                            'dnn_feature_maps','pca_feature_maps', args.dnn, 
                            'pretrained-'+str(args.pretrained), 'layers-'+args.layers)
    file_name = 'pca_feature_maps_'+img_category
    np.save(os.path.join(save_dir, file_name), fmaps_test)
    del fmaps_test