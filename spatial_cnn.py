#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 12 13:20:01 2024

@author: syed
"""
from logger import log
from logger import Logs
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from math import ceil
from sklearn.metrics import r2_score
import torch.nn.functional as F
import os
import configuration as conf
from sklearn.impute import SimpleImputer


PATH = conf.FEATURES_DIRECTORY


def next_batch(batch_size, X_data, y_data):
    idx = np.random.choice(len(X_data), batch_size, replace=False)
    return X_data[idx], y_data[idx]



class ConvNet(nn.Module):
    def __init__(self, nbfilter1, nbfilter2, nbfilter3, shapeconv, shapepool, finalshape, L, input_channels=4):
        super(ConvNet, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=nbfilter1,
            kernel_size=shapeconv,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            in_channels=nbfilter1,
            out_channels=nbfilter2,
            kernel_size=shapeconv,
            padding=1,
        )
        self.conv3 = nn.Conv2d(
            in_channels=nbfilter2,
            out_channels=nbfilter3,
            kernel_size=shapeconv,
            padding=1,
        )
        
        # Adjusted pooling layer
        self.pool = nn.MaxPool2d(kernel_size=shapepool, stride=shapepool, padding=1)

        # Calculate the size of the tensor after the final pooling layer
        self._to_linear = self._get_conv_output((input_channels, L, L))
        
        self.fc1 = nn.Linear(self._to_linear, nbfilter3)
        self.out = nn.Linear(nbfilter3, 1)
        self._initialize_weights()

    def _get_conv_output(self, shape):
        bs = 1
        input = torch.rand(bs, *shape)
        output_feat = self._forward_features(input)
        n_size = output_feat.data.view(bs, -1).size(1)
        return n_size

    def _forward_features(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
        features = F.relu(self.fc1(x))
        outputs = self.out(features)
        return outputs, features


def load_data(rep, country):
    PATH = os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY)
    X_train = np.load(os.path.join(PATH , "cnn_x_pix_train.npy"))
    log(country, "Loading cnn_x_pix_train from: "+ os.path.join(PATH , "cnn_x_pix_train.npy"))
    X_test = np.load(os.path.join(PATH , "cnn_x_pix_test.npy"))
    log(country, "Loading cnn_x_pix_test from: "+ os.path.join(PATH , "cnn_x_pix_test.npy"))
    y_train = np.load(os.path.join(PATH , "cnn_y_pix_train.npy"))
    log(country, "Loading cnn_y_pix_train from: "+ os.path.join(PATH , "cnn_y_pix_train.npy"))
    y_test = np.load(os.path.join(PATH , "cnn_y_pix_test.npy"))
    log(country, "Loading cnn_y_pix_test from: "+ os.path.join(PATH , "cnn_y_pix_test.npy"))
    y_train_com = np.load(os.path.join(PATH , "features_" + rep , "y_train.npy"))
    log(country, "Loading y_train from: "+ os.path.join(PATH , "features_" + rep , "y_train.npy"))
    y_test_com = np.load(os.path.join(PATH , "features_" + rep , "y_test.npy"))
    log(country, "Loading y_test from: "+ os.path.join(PATH , "features_" + rep , "y_test.npy"))

    return X_train, X_test, y_train, y_test, y_train_com, y_test_com


def reshape_data(X_train, X_test, patch_size):
    X_train = X_train.reshape(-1, 4, patch_size, patch_size)
    X_test = X_test.reshape(-1, 4, patch_size, patch_size)
    return X_train, X_test


def initialize_parameters(L):
    hm_epochs = 30
    batch_size = 32
    nbfilter1 = 32
    nbfilter2 = 64
    nbfilter3 = 128
    shapeconv = 3
    shapepool = 2
    finalshape = 1

    return (
        hm_epochs,
        batch_size,
        nbfilter1,
        nbfilter2,
        nbfilter3,
        shapeconv,
        shapepool,
        finalshape,
    )


# =============================================================================#
# Save model and print summary of the model                                    #
# =============================================================================#
def save_model(cnn, best_test_loss_R2, best_ep, country):
    os.makedirs(os.path.join(conf.OUTPUT_DIR, country, "models"), exist_ok=True)
    torch.save({
            'best_ep': best_ep,
            'model_state_dict': cnn.state_dict(),
            'best_test_loss_R2': best_test_loss_R2
        }, os.path.join(conf.OUTPUT_DIR, country, "models", 'cnn_epa.pth'))
    #torch.save(cnn.state_dict(), "./Models/cnn_epa.pth")
    torch.save(cnn, os.path.join(conf.OUTPUT_DIR, country, "models",'cnn_epa_architecture.pth'))
   
    

# Define a function to extract features and predictions
def extract_features_and_predictions(model, data):
    with torch.no_grad():
        features, predictions = model(data)
        features = features.cpu().numpy()
        predictions = predictions.cpu().numpy()
    return np.asarray(features, dtype=np.float32), np.asarray(predictions, dtype=np.float32)


def cnn(rep, r_split, country):
    log(country,"Begin CNN on population and land cover data")

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load data
    X_train, X_test, y_train, y_test, y_train_com, y_test_com = load_data(rep, country)

    # Reshape data
    L = conf.cnn_settings[country]['length']  # width and length of pixel patches
    X_train, X_test = reshape_data(X_train, X_test, L)

    # Initialize parameters
    (
        hm_epochs,
        batch_size,
        nbfilter1,
        nbfilter2,
        nbfilter3,
        shapeconv,
        shapepool,
        finalshape,
    ) = initialize_parameters(L)

    # Print confirmation
    log(country, "Data loading and reshaping complete.")
    log(country, f"Training data shape: {X_train.shape}")
    log(country, f"Test data shape: {X_test.shape}")
    log(country, f"Training Y shape: {y_train.shape}")
    log(country, f"Test Y shape: {y_test.shape}")
    log(country, "Parameters initialized.")

    # Convert numpy arrays to PyTorch tensors and move to the appropriate device
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
    
    #l_model = False
    #if l_model:
    #Initialize the model
    model = ConvNet(
        nbfilter1, nbfilter2, nbfilter3, shapeconv, shapepool, finalshape, L
    ).to(device)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    # Training loop
    best_test_loss_R2 = -float("inf")
    for epoch in range(hm_epochs):
        model.train()
        epoch_loss = 0
        num_batches = len(X_train) // batch_size
        for _ in range(num_batches):
            epoch_x, epoch_y = next_batch(batch_size, X_train_tensor, y_train_tensor)

            optimizer.zero_grad()
            outputs, _ = model(epoch_x)
            loss = criterion(outputs.squeeze(), epoch_y.squeeze())
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            train_pred, _ = model(X_train_tensor)
            test_pred, _ = model(X_test_tensor)

            train_loss = criterion(
                train_pred.squeeze(), y_train_tensor.squeeze()
            ).item()
            test_loss = criterion(test_pred.squeeze(), y_test_tensor.squeeze()).item()

            train_R2 = r2_score(
                y_train_tensor.cpu().numpy(), train_pred.cpu().numpy().squeeze()
            )
            test_R2 = r2_score(
                y_test_tensor.cpu().numpy(), test_pred.cpu().numpy().squeeze()
            )

            log(country, 
                f"Epoch {epoch+1}/{hm_epochs}, Train Loss: {epoch_loss:.6f}, Test Loss: {test_loss:.6f}, Train R^2: {train_R2:.6f}, Test R^2: {test_R2:.6f}"
            )
            if test_R2 > best_test_loss_R2:
                best_test_loss_R2 = test_R2
                best_ep = epoch + 1
                save_model(model, best_test_loss_R2, best_ep, country)
        scheduler.step()
               
    #Utilize the best model 
    model = ConvNet(nbfilter1, nbfilter2, nbfilter3, shapeconv, shapepool, finalshape, L).to(device)
    checkpoint = torch.load(os.path.join(conf.OUTPUT_DIR, country, "models", 'cnn_epa.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set model to evaluation mode
    
    log(country, "Best model restored")

    # Load the data
    PATH = os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY)
    np.load(PATH + "cnn_x_pix_train.npy")
    info_pix_train = np.load(PATH + "cnn_info_pix_train.npy")
    info_pix_test = np.load(PATH + "cnn_info_pix_test.npy")
    info_train = pd.DataFrame(np.load(PATH + "info_train.npy"))
    info_test = pd.DataFrame(np.load(PATH + "info_test.npy"))
    
    # feature calculation and pixel prediction
    FeatPixTrain, PredPixTrain = extract_features_and_predictions(model, X_train_tensor)
    FeatPixTest, PredPixTest = extract_features_and_predictions(model, X_test_tensor)
    
    # merge features and predictions with information (municipality, year) for each pixel
    FeatPixTrain = pd.DataFrame(np.concatenate((info_pix_train, FeatPixTrain), axis=1))
    FeatPixTest = pd.DataFrame(np.concatenate((info_pix_test, FeatPixTest), axis=1))
    PredPixTrain = pd.DataFrame(np.concatenate((info_pix_train, PredPixTrain), axis=1))
    PredPixTest = pd.DataFrame(np.concatenate((info_pix_test, PredPixTest), axis=1))
    
    # group pixel features and predictions by (municipality, year)
    FeatTrain = FeatPixTrain.groupby([0, 1]).agg({key: "mean" for key in range(2, len(FeatPixTrain.columns))}).reset_index()
    FeatTrain = pd.merge(info_train, FeatTrain, how='left', on=[0, 1])
    FeatTrain = FeatTrain.drop([0, 1], axis=1)
    FeatTrain = np.array(FeatTrain)
    os.makedirs(os.path.join(conf.OUTPUT_DIR, country, conf.FEATURES_DIRECTORY,'features_' + rep + '_'+ str(r_split)), exist_ok=True)
    np.save(os.path.join(conf.OUTPUT_DIR, country, conf.FEATURES_DIRECTORY,'features_' + rep + '_'+ str(r_split), "cnn_feat_x_train.npy"), FeatTrain)
    log(country, "cnn_feat_x_train feature saved at:"+ os.path.join(conf.OUTPUT_DIR, country, conf.FEATURES_DIRECTORY,'features_' + rep + '_'+ str(r_split), "cnn_feat_x_train.npy"))
    
    FeatTest = FeatPixTest.groupby([0, 1]).agg({key: "mean" for key in range(2, len(FeatPixTest.columns))}).reset_index()
    FeatTest = pd.merge(info_test, FeatTest, how='left', on=[0, 1])
    FeatTest = FeatTest.drop([0, 1], axis=1)
    FeatTest = np.array(FeatTest)
    np.save(os.path.join(conf.OUTPUT_DIR, country, conf.FEATURES_DIRECTORY, 'features_' + rep + "_" + str(r_split), "cnn_feat_x_test.npy"), FeatTest)
    log(country, "cnn_feat_x_test feature saved at:"+ os.path.join(conf.OUTPUT_DIR, country, conf.FEATURES_DIRECTORY, 'features_' + rep + "_" + str(r_split), "cnn_feat_x_test.npy"))
    

    PredTrain = PredPixTrain.groupby([0, 1]).agg({2:"mean"}).reset_index()
    PredTrain = pd.merge(info_train, PredTrain, how='left', on=[0, 1])
    PredTrain = PredTrain.drop([0, 1], axis=1)
    PredTrain = np.array(PredTrain)
    np.save(os.path.join(conf.OUTPUT_DIR, country, conf.FEATURES_DIRECTORY, 'features_' + rep + "_" + str(r_split), "cnn_pred_train.npy"), PredTrain)
    log(country, "cnn_pred_train feature saved at:"+  os.path.join(conf.OUTPUT_DIR, country, conf.FEATURES_DIRECTORY, 'features_' + rep + "_" + str(r_split), "cnn_pred_train.npy"))
    
    PredTest = PredPixTest.groupby([0, 1]).agg({2:"mean"}).reset_index()
    PredTest = pd.merge(info_test, PredTest, how='left', on=[0, 1])
    PredTest = PredTest.drop([0, 1], axis=1)
    PredTest = np.array(PredTest)
    np.save(os.path.join(conf.OUTPUT_DIR, country, conf.FEATURES_DIRECTORY, 'features_' + rep + "_" + str(r_split), "cnn_pred_test.npy"), PredTest)
    log(country, "cnn_pred_test feature saved at:"+ os.path.join(conf.OUTPUT_DIR, country, conf.FEATURES_DIRECTORY, 'features_' + rep + "_" + str(r_split), "cnn_pred_test.npy"))
    
    #print(f"Length of y_test_com: {len(y_test_com)}")
    #print(f"Length of PredTest: {len(PredTest)}")
    # Create a mask to filter out NaN values from PredTest
    #mask = ~np.isnan(PredTest)
    #print("NAN indices", np.argwhere(np.isnan(PredTest)))
    # Apply the mask to filter out NaN values from both PredTest and y_test_com
    
    #PredTest_clean = np.delete(PredTest, np.argwhere(np.isnan(PredTest)))
    #y_test_com_clean = np.delete(y_test_com, np.argwhere(np.isnan(PredTest)))
    
    imputer = SimpleImputer(strategy='mean')
    PredTest_clean = imputer.fit_transform(PredTest)
    y_test_com_clean = imputer.fit_transform(y_test_com)
    #print(f"Length of y_test_com_clean: {len(y_test_com)}")
    #print(f"Length of PredTest_clean: {len(PredTest_clean)}")
        
    Final_R2 = r2_score(y_test_com_clean, PredTest_clean)
    best_ep = checkpoint['best_ep']
    best_test_loss_R2 = checkpoint['best_test_loss_R2']
    log(country, "Features saved in folder \"" + os.path.join(conf.OUTPUT_DIR, country, conf.FEATURES_DIRECTORY,'features_' + rep + '_'+ str(r_split)) )
    
    log(country, f"Final R2 associate with best loss:  {Final_R2:.6f}, Test R^2: {best_test_loss_R2:.6f} reached at epoch: {best_ep}")
    
    log(country, "End CNN on population and land cover data")
        
        