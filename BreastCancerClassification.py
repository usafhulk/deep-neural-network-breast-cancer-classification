#!/usr/bin/env python
# coding: utf-8

# # <a id='toc1_'></a>[Deep Neural Network for Breast Cancer Classification](#toc0_)
# 

# **Table of contents**<a id='toc0_'></a>    
# - [Deep Neural Network for Breast Cancer Classification](#toc1_)    
#   - [Setup](#toc1_1_)    
#     - [Installing Required Libraries](#toc1_3_1_)    
#   - [Load the Data](#toc1_2_)    
#     - [Breast Cancer Wisconsin (Diagnostic)](#toc1_4_1_)    
#   - [Data Preprocessing](#toc1_3_)    
#   - [Build and Train the Neural Network Model](#toc1_4_)    
#   - [Visualize the Training and Test Loss](#toc1_5_)    
#   - [Exercises](#toc1_6_)    
#     - [Exercise 1 - Change to different optimizer: SGD](#toc1_8_1_)    
#     - [Exercise 2 - Change the number of neurons](#toc1_8_2_)    
#     - [Exercise 3 - Try different dataset - Iris Dataset](#toc1_8_3_)    
#   - [Authors: Christopher Banner](#toc1_7_)    
# 

# # <a id='toc1_1_'></a>[Setup](#roc0)




from ucimlrepo import fetch_ucirepo 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt



# # <a id='toc1_2_'></a>[Loading the Data](#roc0)
# 
# ### Breast Cancer Wisconsin (Diagnostic)
# 
# 
# The [Breast Cancer Wisconsin (Diagnostic) dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) is a classic dataset used for classification tasks. It contains 569 samples of breast cancer cells, each with 30 features. The dataset is divided into two classes: benign and malignant. The goal is to classify the breast cancer cells into one of the two classes.
# 
# This dataset is free to use and is licensed under a Creative Commons Attribution 4.0 International (CC BY 4.0) license.

# In[4]:


from ucimlrepo import fetch_ucirepo  # Import the function to fetch datasets from UCI ML Repo

# fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)  # Fetch the Breast Cancer Wisconsin Diagnostic dataset by its ID

# data (as pandas dataframes) 
X = breast_cancer_wisconsin_diagnostic.data.features  # Extract the features as a pandas DataFrame
y = breast_cancer_wisconsin_diagnostic.data.targets   # Extract the targets as a pandas DataFrame

# print the first few rows of the data
print(X.head())  # Display the first five rows of the features DataFrame

# print the first few rows of the target
print(y.head())  # Display the first five rows of the targets DataFrame


# In[5]:


print(f'X shape: {X.shape}')  # Display the shape of the features DataFrame
print(f'y shape: {y.shape}')  # Display the shape of the targets DataFrame
print(y['Diagnosis'].value_counts())  # Display the count of each unique value in the 'Diagnosis' column of the targets DataFrame


# The dataset is **imbalanced**, with more benign samples than malignant samples. So I will process the data and randomly select 200 samples from each malignant and benign.

# In[6]:


# Combine features and target into a single DataFrame for easier manipulation
data = pd.concat([X, y], axis=1)  # Concatenate features and target DataFrames along columns

# Separate the two classes
data_B = data[data['Diagnosis'] == 'B']  # Select all rows where Diagnosis is 'B' (benign)
data_M = data[data['Diagnosis'] == 'M']  # Select all rows where Diagnosis is 'M' (malignant)

# Select 200 samples from each class
data_B = data_B.sample(n=200, random_state=42)  # Randomly sample 200 benign cases
data_M = data_M.sample(n=200, random_state=42)  # Randomly sample 200 malignant cases

# Combine the two classes
balanced_data = pd.concat([data_B, data_M])  # Concatenate the sampled benign and malignant data

print(balanced_data['Diagnosis'].value_counts())  # Display the count of each class in the balanced dataset


# # <a id='toc1_3_'></a>[Data Preprocessing](#roc0)

# In[7]:


# Separate features and targets
X = balanced_data.drop('Diagnosis', axis=1)  # Remove the 'Diagnosis' column to get features
y = balanced_data['Diagnosis']  # Extract the 'Diagnosis' column as the target

# Convert the targets to binary labels
y = y.map({'B': 0, 'M': 1})  # Map 'B' to 0 and 'M' to 1 for binary classification

print(X)  # Display the features DataFrame
print(y)  # Display the binary target Seriest


# Now I  standardize the feature values using the `StandardScaler` from scikit-learn.
# 
# Standardizing the data involves transforming the features so that they have a mean of 0 and a standard deviation of 1. This helps in ensuring that all features contribute equally to the result and helps the model converge faster during training.
# 
# 1. **Fitting the Scaler**: I calculate the mean and standard deviation for each feature in the training set using the `fit` method of the `StandardScaler`.
# 2. **Transforming the Training Data**: I apply the standardization to the training data using the `transform` method, which scales the features accordingly.
# 3. **Transforming the Test Data**: I apply the same transformation to the test data using the same scaler. This ensures that both training and test sets are standardized in the same way.
# 
# By standardizing the data, I am making sure that each feature contributes equally to the training process, which helps in achieving better performance and faster convergence of the neural network model.
# 
# Converting toNumPy is the last stepnsors.

# In[11]:

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)  # Split the data into 80% train and 20% test, stratified by the target

print(f'X_train shape: {X_train.shape}')  # Display the shape of the training features
print(f'y_train shape: {y_train.shape}')  # Display the shape of the training targets
print(f'X_test shape: {X_test.shape}')    # Display the shape of the test features
print(f'y_test shape: {y_test.shape}')    # Display the shape of the test targets

#  Standardize the data
# Initialize the StandardScaler
scaler = StandardScaler()  # Create a StandardScaler instance

# Fit the scaler on the training data and transform it
X_train = scaler.fit_transform(X_train)  # Fit to training data and transform

# Transform the test data using the same scaler
X_test = scaler.transform(X_test)  # Transform test data using the fitted scaler

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)  # Convert training features to float32 tensor
X_test = torch.tensor(X_test, dtype=torch.float32)    # Convert test features to float32 tensor
y_train = torch.tensor(y_train.values, dtype=torch.long)  # Convert training targets to long tensor
y_test = torch.tensor(y_test.values, dtype=torch.long)    # Convert test targets to long tensor

# Create DataLoader for training and test sets
train_dataset = TensorDataset(X_train, y_train)  # Create a TensorDataset for training data
test_dataset = TensorDataset(X_test, y_test)     # Create a TensorDataset for test data

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)   # DataLoader for training data
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)    # DataLoader for test data



# ##### Below I will start defining the network architecuter and training the model.<br/>
# ##### #`nn.Module` from PyTorch will be used and the output layer will contain 2 neurons (corresponding with the two classes I have)<br/>
# <img src="Images/8-8-2.jpg" alt="image" width="50%">

# In[13]:


class ClassificationNet(nn.Module):
    def __init__(self, input_units=30, hidden_units=64, output_units=2):
        super(ClassificationNet, self).__init__()  # Call the parent class constructor
        self.fc1 = nn.Linear(input_units, hidden_units)  # First fully connected layer
        self.fc2 = nn.Linear(hidden_units, output_units)  # Output layer

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # Apply ReLU activation to the first layer
        x = self.fc2(x)  # Output layer (no activation, as CrossEntropyLoss expects raw logits)
        return x

# Instantiate the model
model = ClassificationNet(input_units=30, hidden_units=64, output_units=2)  # Create an instance of the network

print(model)  # Print the model architecture

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Use cross-entropy loss for classification
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Use Adam optimizer with learning rate 0.001


# In[14]:


epochs = 10  # Number of training epochs
train_losses = []  # List to store training loss for each epoch
test_losses = []   # List to store test loss for each epoch

for epoch in range(epochs):
    # Training phase
    model.train()  # Set the model to training mode
    running_loss = 0.0  # Initialize running loss for this epoch
    for X_batch, y_batch in train_loader:  # Iterate over batches in the training loader
        optimizer.zero_grad()  # Zero the gradients
        outputs = model(X_batch)  # Forward pass
        loss = criterion(outputs, y_batch)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model parameters
        running_loss += loss.item()  # Accumulate loss

    train_loss = running_loss / len(train_loader)  # Average training loss for this epoch
    train_losses.append(train_loss)  # Store training loss

    # Evaluation phase on test set
    model.eval()  # Set the model to evaluation mode
    test_loss = 0.0  # Initialize test loss
    with torch.no_grad():  # Disable gradient computation for evaluation
        for X_batch, y_batch in test_loader:  # Iterate over batches in the test loader
            test_outputs = model(X_batch)  # Forward pass
            loss = criterion(test_outputs, y_batch)  # Compute loss
            test_loss += loss.item()  # Accumulate test loss

    test_loss /= len(test_loader)  # Average test loss for this epoch
    test_losses.append(test_loss)  # Store test loss

    print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')  # Print losses for this epoch


# # <a id='toc1__'></a>[Visualize the Training and Test Loss](#toc0_)

# In[15]:


# Plot the loss curves
plt.figure(figsize=(10, 6))  # Set the figure size
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')  # Plot training loss
plt.plot(range(1, epochs + 1), test_losses, label='Test Loss', linestyle='--')  # Plot test loss with dashed line
plt.xlabel('Epoch')  # Label for x-axis
plt.ylabel('Loss')  # Label for y-axis
plt.title('Training and Test Loss Curve')  # Title of the plot
plt.legend()  # Show legend
plt.grid(True)  # Show grid
plt.show()  # Display the plot

