import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from keras.datasets import mnist
# if torch.cuda.is_available():
#     print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
# else:
#     print("CUDA is not available. Using CPU.")
def gradient(data,target,w):
    device="cuda"
    model = nn.Sequential(
    nn.Linear(784,64),    #Input layer with 64 features, hidden layer with 60 units
    nn.ReLU(),            #Activation for the first linear layer
    nn.Linear(64, 10),    # Output layer with 10 units (for 10 classes)
    ).to(device=device)
    w=torch.tensor(w).to(device=device) 
    #GETTING WEIGHTS AND BIASES FOR THE FIRST LINEAR LAYER
    custom_weights_fc1 = torch.tensor(torch.reshape(w[0:50176],(64,784)),dtype=torch.float32).to(device=device)   # Weight matrix for Linear(10, 5)
    custom_bias_fc1 = torch.tensor(w[50176:50240], dtype=torch.float32).to(device=device)           # Bias vector for Linear(10, 5)

    ##GETTING WEIGHTS AND BIASES FOR THE SECOND LINEAR LAYER
    custom_weights_fc2 = torch.tensor(torch.reshape(w[50240:50880],(10,64)), dtype=torch.float32).to(device=device)    # Weight matrix for Linear(5, 1)
    custom_bias_fc2 = torch.tensor(w[50880:50890], dtype=torch.float32).to(device=device)          # Bias vector for Linear(5, 1)

    #TRAINING THE MODEL WITH THESE WEIGHTS
    with torch.no_grad():
        model[0].weight.copy_(custom_weights_fc1).to(device=device) 
        model[0].bias.copy_(custom_bias_fc1).to(device=device) 
        model[2].weight.copy_(custom_weights_fc2).to(device=device) 
        model[2].bias.copy_(custom_bias_fc2).to(device=device) 
    criterion = nn.CrossEntropyLoss()       #Choice of loss function

    #Using the given data to calculate loss
    inputs = torch.tensor(data, dtype=torch.float32).to(device=device) 
    targets = torch.tensor(target, dtype=torch.long).to(device=device)  
    outputs = model(inputs).to(device=device)     #Getting predictions
    loss = criterion(outputs, targets) #Computing loss on the given data
    loss.backward()     #Backward pass to get gradients
    #GETTING OUTPUT AS A NUMPY ARRAY
    grad_list = [param.grad.flatten() for param in model.parameters() if param.requires_grad and param.grad is not None]
    all_grads = torch.cat(grad_list)  # Concatenate all gradients
    grad_array = all_grads.detach().cpu().numpy()  # Convert to NumPy array
    return grad_array,loss.detach().cpu().numpy()