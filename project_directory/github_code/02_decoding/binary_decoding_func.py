"""Binary decoding based on image properties"""

def feature_select(args, sub):
    import statistics as st
    import pandas as pd
    import numpy as np
    import os

    THINGS_dir = os.path.join(args.project_dir, 'eeg_dataset', 'wake_data', 'THINGS')
    
    # load THINGS image metadata
    img_meta_dir = os.path.join(THINGS_dir, 'objectProperties_meanRatings.tsv')
    img_data = pd.read_csv(img_meta_dir, delimiter='\t')

    # Load THINGS training images and concepts names
    img_training_dir  = os.path.join(THINGS_dir, 'image_set', 'training_images')
    concepts_dir = os.listdir(img_training_dir)
    concepts = [concept[6:] for concept in concepts_dir]

    img_data = img_data[img_data['uniqueID'].isin(concepts)]
    img_data = img_data.reset_index()

    # Select object property
    positive_index = []
    negative_index = []
    median = st.median(img_data[args.obj_prop])

    for index, row in img_data.iterrows():
        if row[args.obj_prop] > median:
            positive_index.append((index))
        else:
            negative_index.append((index))
            
    # Import THINGS data
    THINGS_prepr_dir = os.path.join(THINGS_dir, 
                                    'preprocessed_data','sub-'+format(sub,'02'),
                                    'preprocessed_eeg_training.npy')
    THINGS_eeg_data = np.load(THINGS_prepr_dir, allow_pickle=True).item()
    
    # Notations
    No_p = len(positive_index)*10
    No_n = len(negative_index)*10
    repetitions = THINGS_eeg_data['preprocessed_eeg_data'].shape[1]
    channels = THINGS_eeg_data['preprocessed_eeg_data'].shape[2]
    times = int(THINGS_eeg_data['preprocessed_eeg_data'].shape[3]*0.8)
    
    positive_data = np.empty((No_p, repetitions, channels, times))
    negative_data = np.empty((No_n, repetitions, channels, times))

    for i, (p, n) in enumerate(zip(positive_index, negative_index)): 
        positive_data[i*10:i*10+10,:,:,:] = THINGS_eeg_data['preprocessed_eeg_data'][int(p)*10:int(p)*10+10,:,:,20:100]
        negative_data[i*10:i*10+10,:,:,:] = THINGS_eeg_data['preprocessed_eeg_data'][int(n)*10:int(n)*10+10,:,:,20:100]

    positive_data = positive_data.reshape((No_p*repetitions, channels, times))
    negative_data = negative_data.reshape((No_n*repetitions, channels, times))
    
    X = np.empty(((No_p+No_n)*repetitions, channels, times))
    X[:No_p*repetitions,:,:] = positive_data
    X[No_p*repetitions:,:,:] = negative_data

    y_1 = [1]*No_p*repetitions
    y_2 = [0]*No_n*repetitions
    y = np.concatenate((y_1, y_2), axis=0)
    
    return X, y


def binary_classifier(args, X, Y):
    import os
    import numpy as np
    
    DEED_prepr_dir = os.path.join(args.project_dir, 'eeg_dataset','dream_data', 
                                  'preprocessed_data', 'DEED', args.PSG_number)
    DEED_eeg_data = np.load(os.path.join(DEED_prepr_dir, 
                                         args.PSG_number+'_'+args.rem+'.npy'), 
                            allow_pickle=True).item()
    DEED_eeg_data['preprocessed_eeg_data'].shape
    
    from matplotlib import pyplot as plt
    from tqdm import tqdm
    from sklearn.linear_model import LinearRegression

    for t in tqdm(range(80), desc=f'{args.obj_prop} decoding training'):
        reg = LinearRegression().fit(X[:,:,t], Y)
    
    # Notations
    channels = DEED_eeg_data['preprocessed_eeg_data'].shape[0]
    times = DEED_eeg_data['preprocessed_eeg_data'].shape[1]
    
    result = np.empty((times))
    for T in range(times):
        result[T] = reg.predict(DEED_eeg_data['preprocessed_eeg_data'][:,T].reshape((1,channels)))

    # plot the result
    plt.figure()
    plt.plot(range(times), result)
    plt.xlabel('Time (10ms)')
    plt.ylabel('tendency')
    plt.title(args.obj_prop)
    
    # save the result
    save_dir = os.path.join(args.project_dir, 'statistical_result', 'decoding',
                            'binary_decoding', args.PSG_number, args.PSG_number+'_'+args.rem)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    obj_prop = args.obj_prop.replace('.','_')
    plt.savefig(os.path.join(save_dir, 'binary classification of '+obj_prop))
    
    