from logger import log
import numpy as np
import configuration as conf
import torch
import torch.nn as nn
import os
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.metrics import r2_score, accuracy_score
#from torchinfo import summary
import math
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import StandardScaler
from visualization import save_classification_map 



# pip install torchinfo

hm_epochs = 1000
batch_size = 32

timesteps = 14
nb_hidden = 128
num_layers = 1
learning_rate = 0.01
num_classes = 3
best_test_loss = float("inf")
best_test_R2 = 0
best_ep = 0
early_stopping_patience = 100

# =============================================================================#
# Load data X, Y, and w from the preprocessed data                             #
# =============================================================================#
def load_data(rep, r_split, country):
    # Load data
    X_train = np.load(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "timeseries_x_train.npy"))
    log(country, "Loading X_train from: " + os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "timeseries_x_train.npy"))
    X_test = np.load(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "timeseries_x_test.npy"))
    log(country, "Loading X_test from: " + os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "timeseries_x_test.npy"))
    y_train = np.load(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "features_" + rep + "/y_train.npy"))
    log(country, "Loading y_train from: " + os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "features_" + rep + "/y_train.npy"))
    y_test = np.load(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "features_" + rep + "/y_test.npy"))
    log(country,"Loading y_test from: " + os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "features_" + rep + "/y_test.npy"))
    w_train = np.load(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "w_train.npy"))
    log(country, "Loading w_train from: " + os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "w_train.npy"))
    w_test = np.load(os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "w_test.npy"))
    log(country, "Loading w_test from: " + os.path.join(conf.PREPROCESS_DATA_DIR, country, conf.FEATURES_DIRECTORY, "w_test.npy"))

    return X_train, X_test, y_train, y_test, w_train, w_test


# =============================================================================#
# Define neural network architecture                                           #
# =============================================================================#
# def create_model(nb_inputs, timesteps, nb_hidden, num_layers):
#    lstm = nn.LSTM(input_size=nb_inputs, hidden_size=nb_hidden, num_layers=num_layers, batch_first=True)
#    linear = nn.Linear(nb_hidden, 1)
#    return lstm, linear


class LSTMModel(nn.Module):
    def __init__(self, input_size, algorithm):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=nb_hidden, num_layers=num_layers, batch_first=True)
       
        if algorithm == 'classification':
            #print('Classification selected')
            self.fc = nn.Linear(nb_hidden, num_classes)
        else:
            print('Regression selected')
            self.fc = nn.Linear(nb_hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out


# =============================================================================#
# Training function                                                            #
# =============================================================================#
def train_model(algorithm, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_targets = []
    for inputs, targets, weights in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)  # Outputs should be of shape (N, C)
        targets = targets.view(-1)  # Ensure targets are of shape (N,)
        if algorithm == 'classification':
            
            adjusted_targets = targets - 1
            loss = criterion(outputs, adjusted_targets)
        else:
            #print("Outputs Shape train", outputs.squeeze().shape, outputs.squeeze())
            #print("targets Shape train", targets.shape, targets)
            loss = criterion(outputs.squeeze(), targets)
        weighted_loss = (loss * weights.view(-1)).mean()  # Ensure weights are of shape (N,)
        weighted_loss.backward()
        optimizer.step()
        running_loss += weighted_loss.item() * inputs.size(0)
        total_samples += inputs.size(0)
        if algorithm == 'classification':
            predictions = torch.argmax(outputs, dim=1).cpu().numpy() + 1
        else:
            predictions = outputs.squeeze().detach().cpu().numpy()
        targets_np = targets.squeeze().detach().cpu().numpy()
        all_predictions.extend(predictions)
        all_targets.extend(targets_np)
    
    epoch_loss = running_loss / total_samples
    if algorithm == 'classification':
        score = accuracy_score(all_targets, all_predictions)
    else:
        score = r2_score(all_targets, all_predictions)

    return epoch_loss, score, all_targets, all_predictions



# =============================================================================#
# Evaluation function                                                          #
# =============================================================================#
def evaluate_model(algorithm, model, test_loader, criterion):
    model.eval()
    all_predictions = []
    all_targets = []
    running_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets, weights in test_loader:
            outputs = model(inputs)
            targets = targets.view(-1)  # Ensure targets are of shape (N,)
            if algorithm == 'classification':
                adjusted_targets = targets - 1
                loss = criterion(outputs, adjusted_targets)
            else:
                #print("Outputs Shape", outputs.squeeze().shape, outputs.squeeze())
                #print("targets Shape", targets.shape, targets)
                loss = criterion(outputs.squeeze(), targets)
            weighted_loss = (loss * weights).mean()
            running_loss += weighted_loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            if algorithm == 'classification':
                predictions = torch.argmax(outputs, dim=1).cpu().numpy()  + 1
            else:
                predictions = outputs.squeeze().detach().cpu().numpy()
            targets_np = targets.squeeze().detach().cpu().numpy()
            all_predictions.extend(predictions)
            all_targets.extend(targets_np)
    
    epoch_loss = running_loss / total_samples
    if algorithm == 'classification':
        score = accuracy_score(all_targets, all_predictions)
    else:
        score = r2_score(all_targets, all_predictions)

    return epoch_loss, score, all_targets, all_predictions


# =============================================================================#
# Save model and print summary of the model                                                                #
# =============================================================================#
def save_model(country, rep, lstm, best_test_loss, best_test_R2, best_ep):
    os.makedirs(os.path.join(conf.OUTPUT_DIR, country, "models", rep), exist_ok=True)
    torch.save({
            
            'best_ep': best_ep,
            'best_test_loss': best_test_loss,
            'model_state_dict': lstm.state_dict(),
            'best_test_R2': best_test_R2
        }, os.path.join(conf.OUTPUT_DIR, country, "models", rep,'lstm_epa.pth') )
    torch.save(lstm, os.path.join(conf.OUTPUT_DIR, country, "models", rep, 'lstm_epa_architecture.pth'))
    # summary(lstm, input_size=(batch_size, nb_inputs))

#Save Results
def save_results(country, algorithm, rep, test_targets, test_predictions):
    data = pd.read_excel(os.path.join(
        conf.DATA_DIRECTORY, country, conf.RESPONSE_FILE[country]))
    years= list(data[conf.TEMPORAL_GRANULARITY[country]].unique())
    data = data[data[conf.TEMPORAL_GRANULARITY[country]] == max(years)]
    data = data[conf.SPATIAL_TEMPORAL_GRANULARITY[country]]
    os.makedirs(os.path.join(conf.OUTPUT_DIR, country, "results", algorithm), exist_ok=True)
    output = pd.DataFrame({'label': test_targets, 'prediction': test_predictions})
    data = data.reset_index(drop=True)
    results =  pd.concat([data, output], axis=1)
    results.to_excel(os.path.join(conf.OUTPUT_DIR, country, "results", algorithm,  rep + '.xlsx'), index=False)
    if algorithm == 'classification':
        save_classification_map(country, algorithm, rep, max(years))

# =============================================================================#
# Main function                                                                #
# =============================================================================#
def timeseries_lstm(rep, algorithm, r_split, country):
    log(country, "Begin time-series data learning through LSTM model")

    X_train, X_test, y_train, y_test, w_train, w_test = load_data(rep, r_split, country)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create datasets and dataloaders
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long if algorithm == 'classification' else torch.float32),
        torch.tensor(w_train, dtype=torch.float32),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long if algorithm == 'classification' else torch.float32),
        torch.tensor(w_test, dtype=torch.float32),
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize model, criterion, optimizer, and scheduler
    if os.path.exists(os.path.join(conf.OUTPUT_DIR, country, "models",rep, 'lstm_epa.pth')):
        log(country, "Best LSTM Model Loaded at : "+ os.path.join(conf.OUTPUT_DIR, country, "models",rep, 'lstm_epa.pth'))
        checkpoint = torch.load(os.path.join(conf.OUTPUT_DIR, country, "models", rep,'lstm_epa.pth'))
        model = LSTMModel(input_size=X_train.shape[1], algorithm=algorithm)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_ep = checkpoint['best_ep']
        best_test_R2 = checkpoint['best_test_R2']
    else:
        model = LSTMModel(input_size=X_train.shape[1], algorithm=algorithm)
    criterion = nn.CrossEntropyLoss() if algorithm == 'classification' else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #scheduler = StepLR(optimizer, step_size=14, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    best_test_R2 = -float("inf")
    best_test_loss = float("inf")
    best_epoch = 0

    # Training loop
    for epoch in range(hm_epochs):
        # Train the model
        train_loss, train_score, train_targets, train_predictions = train_model(algorithm, model, train_loader, criterion, optimizer)
        
        # Evaluate the model
        test_loss, test_score, test_targets, test_predictions = evaluate_model(algorithm, model, test_loader, criterion)

        # Logging and saving best model
        log(country, f"Epoch {epoch+1}/{hm_epochs}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}, Train Score: {train_score:.6f}, Test Score: {test_score:.6f}")
    
        # Save the best model based on the test score
        if (algorithm == 'classification' and test_score > best_test_R2) or (algorithm != 'classification' and test_score > best_test_R2 and test_loss < best_test_loss):
            best_test_loss = test_loss
            best_test_R2 = test_score
            best_epoch = epoch + 1
            save_model(country, rep, model, best_test_loss, best_test_R2, best_epoch)
            save_results(country, algorithm, rep, test_targets, test_predictions)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if (epochs_no_improve >= early_stopping_patience) and best_test_R2 > 0:
            log(country, f'Early stopping at epoch {epoch+1}')
            break
        
        # Adjust learning rate scheduler
        scheduler.step(test_loss)
    log(country, "Best LSTM Model Saved at : "+ os.path.join(conf.OUTPUT_DIR, country, "models",rep, 'lstm_epa.pth'))
    checkpoint = torch.load(os.path.join(conf.OUTPUT_DIR, country, "models", rep,'lstm_epa.pth'))
    best_ep = checkpoint['best_ep']
    best_test_R2 = checkpoint['best_test_R2']
    log(country, f"Test R2 associate with best Score: {best_test_R2:.6f}  reached at epoch: {best_ep}")
    
    
    # Instantiate your LSTM model
    
    # lstm_saved = LSTMModel(nb_inputs)
    
    
    # # Load the saved model
    # checkpoint = torch.load(os.path.join(conf.OUTPUT_DIR, country, "models",'lstm_epa.pth'))
    # lstm_saved.load_state_dict(checkpoint['model_state_dict'])
    # lstm_saved.eval()
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # lstm_saved.to(device)    
    # with torch.no_grad():
    #     log(country, "Best Model features are saving ...\n")
    #     os.makedirs(os.path.join(conf.OUTPUT_DIR, country, conf.FEATURES_DIRECTORY,'features_' + rep + '_'+ str(r_split)), exist_ok=True)
    #     #log(country, "Features saved at: "+ os.path.join(conf.OUTPUT_DIR, country, conf.FEATURES_DIRECTORY,'features_' + rep + '_'+ str(r_split)))
    #     X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    #     X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    #     # Get features and predictions for training data
    #     trained_predictions, trained_features = lstm_saved(X_train_tensor)

    #     # Get features and predictions for test data
    #     test_predictions, test_features = lstm_saved(X_test_tensor)
    #     log(country, 'Type of test_predictions is:'+str(type(test_predictions)))
    #     log(country, "X_test_tensor is :"+str(y_test))
    #     test_predictions = test_predictions.numpy()
    #     test_predictions = [int(round_(num,2)) for num in test_predictions]
    #     log(country, "test_predictions is :"+str(test_predictions))
        
    #     for i, feature in enumerate(trained_features):
    #         log(country, f"trained features {i} Shape: {feature.shape}")
    #     for i, feature in enumerate(test_features):
    #         log(country, f"test features {i} Shape: {feature.shape}")
    #     log(country, "trained predictions Shape:"+ str(trained_predictions.shape))
    #     log(country, "test predictions Shape:"+ str(test_predictions.shape))
        
    #     trained_features = np.stack((trained_features[0].numpy(), trained_features[1].numpy()), axis=0)
    #     test_features = np.stack((test_features[0].numpy(), test_features[1].numpy()), axis=0)
        
    #     np.save(os.path.join(conf.OUTPUT_DIR, country, conf.FEATURES_DIRECTORY,'features_' + rep + '_'+ str(r_split), 'lstm_feat_x_train.npy'), trained_features)
    #     log(country, "trained feature saved: "+ os.path.join(conf.OUTPUT_DIR, country, conf.FEATURES_DIRECTORY,'features_' + rep + '_'+ str(r_split), 'lstm_feat_x_train.npy'))
    #     np.save(os.path.join(conf.OUTPUT_DIR, country, conf.FEATURES_DIRECTORY,'features_' + rep + '_'+ str(r_split),  'lstm_feat_x_test.npy'), test_features)
    #     log(country, "test feature saved: "+ os.path.join(conf.OUTPUT_DIR, country, conf.FEATURES_DIRECTORY,'features_' + rep + '_'+ str(r_split),  'lstm_feat_x_test.npy'))
    #     np.save(os.path.join(conf.OUTPUT_DIR, country, conf.FEATURES_DIRECTORY,'features_' + rep + '_'+ str(r_split),  'lstm_pred_train.npy'), trained_predictions)
    #     log(country, "trained predictions saved: "+ os.path.join(conf.OUTPUT_DIR, country, conf.FEATURES_DIRECTORY,'features_' + rep + '_'+ str(r_split),  'lstm_pred_train.npy'))
    #     np.save(os.path.join(conf.OUTPUT_DIR, country, conf.FEATURES_DIRECTORY,'features_' + rep + '_'+ str(r_split),  'lstm_pred_test.npy'), test_predictions)
    #     log(country, "test predictions saved: "+ os.path.join(conf.OUTPUT_DIR, country, conf.FEATURES_DIRECTORY,'features_' + rep + '_'+ str(r_split),  'lstm_pred_test.npy'))
        
    # #trained_features_np = trained_features.numpy()
    # #test_features_np = test_features.numpy() 
    # best_ep = checkpoint['best_ep']
    # best_test_loss_R2 = checkpoint['best_test_loss_R2']
    # log(country, "Features saved in folder \"" + os.path.join(conf.OUTPUT_DIR, country, conf.FEATURES_DIRECTORY,'features_' + rep + '_'+ str(r_split)) + "\"")
    # log(country, f"Test R2 associate with best loss: {best_test_loss_R2:.6f}  reached at epoch: {best_ep}")

    log(country, "End time-series data learning through LSTM model")
