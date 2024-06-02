from logger import log
from logger import Logs
import numpy as np
import configuration as conf
import torch
import torch.nn as nn
import os
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from torchinfo import summary


# pip install torchinfo

hm_epochs = 250
batch_size = 64
nb_inputs = 70
timesteps = 14
nb_hidden = 128
num_layers = 1
learning_rate = 0.01

# Initialize best test loss and R squared
best_test_loss = float("inf")
best_test_loss_R2 = 0
best_ep = 0


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
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=nb_inputs,
            hidden_size=nb_hidden,
            num_layers=num_layers,
            batch_first=True,
        )
        
        self.fc = nn.Linear(nb_hidden, 1)

    def forward(self, x):

        # Forward propagate LSTM
        #out, _ = self.lstm(x)

        # Apply linear transformation to the entire sequence
        #out = self.fc(out)
        
        #h0 = torch.zeros(num_layers, x.size(0), nb_hidden).to(x.device)
        #c0 = torch.zeros(num_layers, x.size(0), nb_hidden).to(x.device)

        out,features = self.lstm(x)
        
        out = self.fc(out)
        return out, features


# =============================================================================#
# Training function                                                            #
# =============================================================================#
def train_model(model, train_loader, criterion, lstm_optimizer):
    lstm = model
    lstm.train()
    # linear.train()
    running_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_targets = []
    for inputs, targets, weights in train_loader:
        lstm_optimizer.zero_grad()
        outputs,_ = lstm(inputs)
        loss = criterion(outputs.squeeze(), targets.view(-1))
        weighted_loss = (loss * weights).mean()
        weighted_loss.backward()
        lstm_optimizer.step()
        running_loss += weighted_loss.item() * inputs.size(0)
        total_samples += inputs.size(0)

        # Convert predictions and targets to numpy arrays
        predictions = outputs.squeeze().detach().cpu().numpy()
        targets_np = targets.squeeze().detach().cpu().numpy()

        # Collect predictions and targets for computing R2 later
        all_predictions.extend(predictions)
        all_targets.extend(targets_np)
    r2 = r2_score(all_targets, all_predictions)
    epoch_loss = running_loss / total_samples
    return epoch_loss, r2


# =============================================================================#
# Evaluation function                                                          #
# =============================================================================#
def evaluate_model(model, test_loader, criterion):
    lstm = model
    lstm.eval()
    all_predictions = []
    all_targets = []
    running_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for inputs, targets, weights in test_loader:
            outputs,_ = lstm(inputs)
            outputs = outputs.flatten()
            # predictions.extend(outputs.squeeze().tolist())
            loss = criterion(outputs.squeeze(), targets.view(-1))
            weighted_loss = (loss * weights).mean()
            running_loss += weighted_loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

            # Convert predictions and targets to numpy arrays
            predictions = outputs.squeeze().detach().cpu().numpy()
            targets_np = targets.squeeze().detach().cpu().numpy()

            # Collect predictions and targets for computing R2 later
            all_predictions.extend(predictions)
            all_targets.extend(targets_np)
    # Compute R2 value
    r2 = r2_score(all_targets, all_predictions)

    # Compute epoch loss
    epoch_loss = running_loss / total_samples
    return epoch_loss, r2


# =============================================================================#
# Save model and print summary of the model                                                                #
# =============================================================================#
def save_model(country, lstm, best_test_loss, best_test_loss_R2, best_ep):
    os.makedirs(os.path.join(conf.OUTPUT_DIR, country, "models"), exist_ok=True)
    torch.save({
            'best_ep': best_ep,
            'best_test_loss': best_test_loss,
            'model_state_dict': lstm.state_dict(),
            'best_test_loss_R2': best_test_loss_R2
        }, os.path.join(conf.OUTPUT_DIR, country, "models",'lstm_epa.pth') )
    torch.save(lstm, os.path.join(conf.OUTPUT_DIR, country, "models",'lstm_epa_architecture.pth'))
    # summary(lstm, input_size=(batch_size, nb_inputs))



# =============================================================================#
# Main function                                                                #
# =============================================================================#
def timeseries_lstm(rep, r_split, country):
    print("Begin time-series data learning through LSTM model")

    # Load data
    X_train, X_test, y_train, y_test, w_train, w_test = load_data(rep, r_split, country)

    # Prepare your data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(w_train, dtype=torch.float32),
    )

    test_dataset = TensorDataset(
        torch.tensor(X_test_scaled, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
        torch.tensor(w_test, dtype=torch.float32),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize the model, loss function, and optimizer
    lstm = LSTMModel()
    criterion = nn.MSELoss()
    lstm_optimizer = optim.Adam(lstm.parameters(), lr=learning_rate)
    best_test_loss_R2 = 0
    best_test_loss = 0
    best_epoch = 0
    # Training loop
    for epoch in range(hm_epochs):
        train_loss, train_R2 = train_model(
            lstm, train_loader, criterion, lstm_optimizer
        )
        test_loss, test_R2 = evaluate_model(lstm, test_loader, criterion)
        log(country,
            f"Epoch {epoch+1}/{hm_epochs}, train_loss: {train_loss:.6f}, 'test_loss:',{test_loss:.6f}, 'train_R^2:',{train_R2:.6f}, 'test_R^2:',{test_R2:.6f}"
        )
        if (test_R2 > best_test_loss_R2) & (test_R2 < train_R2):
            
            best_test_loss = test_loss
            best_test_loss_R2 = test_R2
            best_epoch = epoch + 1
            save_model(country, lstm, best_test_loss, best_test_loss_R2, best_epoch)
    log(country, "Best LSTM Model Saved at : "+ os.path.join(conf.OUTPUT_DIR, country, "models",'lstm_epa.pth'))
    # Instantiate your LSTM model
    
    lstm_saved = LSTMModel()
    
    
    # Load the saved model
    checkpoint = torch.load(os.path.join(conf.OUTPUT_DIR, country, "models",'lstm_epa.pth'))
    lstm_saved.load_state_dict(checkpoint['model_state_dict'])
    lstm_saved.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lstm_saved.to(device)    
    with torch.no_grad():
        log(country, "Best Model features are saving ...\n")
        os.makedirs(os.path.join(conf.OUTPUT_DIR, country, conf.FEATURES_DIRECTORY,'features_' + rep + '_'+ str(r_split)), exist_ok=True)
        #log(country, "Features saved at: "+ os.path.join(conf.OUTPUT_DIR, country, conf.FEATURES_DIRECTORY,'features_' + rep + '_'+ str(r_split)))
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        # Get features and predictions for training data
        trained_predictions, trained_features = lstm_saved(X_train_tensor)

        # Get features and predictions for test data
        test_predictions, test_features = lstm_saved(X_test_tensor)
        
        for i, feature in enumerate(trained_features):
            log(country, f"trained features {i} Shape: {feature.shape}")
        for i, feature in enumerate(test_features):
            log(country, f"test features {i} Shape: {feature.shape}")
        log(country, "trained predictions Shape:"+ str(trained_predictions.shape))
        log(country, "test predictions Shape:"+ str(test_predictions.shape))
        
        trained_features = np.stack((trained_features[0].numpy(), trained_features[1].numpy()), axis=0)
        test_features = np.stack((test_features[0].numpy(), test_features[1].numpy()), axis=0)
        
        np.save(os.path.join(conf.OUTPUT_DIR, country, conf.FEATURES_DIRECTORY,'features_' + rep + '_'+ str(r_split), 'lstm_feat_x_train.npy'), trained_features)
        log(country, "trained feature saved: "+ os.path.join(conf.OUTPUT_DIR, country, conf.FEATURES_DIRECTORY,'features_' + rep + '_'+ str(r_split), 'lstm_feat_x_train.npy'))
        np.save(os.path.join(conf.OUTPUT_DIR, country, conf.FEATURES_DIRECTORY,'features_' + rep + '_'+ str(r_split),  'lstm_feat_x_test.npy'), test_features)
        log(country, "test feature saved: "+ os.path.join(conf.OUTPUT_DIR, country, conf.FEATURES_DIRECTORY,'features_' + rep + '_'+ str(r_split),  'lstm_feat_x_test.npy'))
        np.save(os.path.join(conf.OUTPUT_DIR, country, conf.FEATURES_DIRECTORY,'features_' + rep + '_'+ str(r_split),  'lstm_pred_train.npy'), trained_predictions)
        log(country, "trained predictions saved: "+ os.path.join(conf.OUTPUT_DIR, country, conf.FEATURES_DIRECTORY,'features_' + rep + '_'+ str(r_split),  'lstm_pred_train.npy'))
        np.save(os.path.join(conf.OUTPUT_DIR, country, conf.FEATURES_DIRECTORY,'features_' + rep + '_'+ str(r_split),  'lstm_pred_test.npy'), test_predictions)
        log(country, "test predictions saved: "+ os.path.join(conf.OUTPUT_DIR, country, conf.FEATURES_DIRECTORY,'features_' + rep + '_'+ str(r_split),  'lstm_pred_test.npy'))
        
    #trained_features_np = trained_features.numpy()
    #test_features_np = test_features.numpy() 
    best_ep = checkpoint['best_ep']
    best_test_loss_R2 = checkpoint['best_test_loss_R2']
    log(country, "Features saved in folder \"" + os.path.join(conf.OUTPUT_DIR, country, conf.FEATURES_DIRECTORY,'features_' + rep + '_'+ str(r_split)) + "\"")
    log(country, f"Test R2 associate with best loss: {best_test_loss_R2:.6f}  reached at epoch: {best_ep}")

    log(country, "End time-series data learning through LSTM model")
