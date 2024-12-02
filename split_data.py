import numpy as np
from keras.datasets import mnist
def split_data(n,d):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()  #Loading data
    x_train = x_train.reshape((x_train.shape[0], 28 * 28))   #Flattening data
    d=np.array(d)
    data = x_train
    target=y_train
    k=len(data)
    d=d/k    # Define the ratios
    num_samples = data.shape[0]
    indices = np.cumsum([int(num_samples * ratio) for ratio in d]) #calculate the indices for the splits
    #CHECK IF THE NUMBER OF RATIOS PROVIDED ARE ENOUGH FOR THE DATA SPLIT REQUESTED
    if len(indices) != n - 1:
        raise ValueError("Number of ratios provided must be one less than the number of splits.")
    #SPLIT THE DATA
    split_data_list = []
    split_target_list = []
    start_index = 0
    for end_index in indices:
        split_data_list.append(data[start_index:end_index])
        split_target_list.append(target[start_index:end_index])
        start_index = end_index
    split_data_list.append(data[start_index:])
    split_target_list.append(target[start_index:])
    return split_data_list,split_target_list