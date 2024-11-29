import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score, roc_curve
from scipy.optimize import minimize
from scipy.special import expit as sigmoid
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from pygam import LogisticGAM, s


#######################DWLR#################################
# Activation functions
def relu(z):
    return np.maximum(0, z)

def tanh(z):
    return np.tanh(z)

def softplus(z):
    return np.log(1 + np.exp(z))

# Custom logistic regression with interaction terms and dynamic weights
def custom_logistic_loss_3_predictors(params, x1, x2, x3, y, activation):
    params_a = params[:4]
    params_b = params[4:8]
    params_c = params[8:12]
    params_inter = params[12:]
    
    # Compute weights using the activation function
    a_t = activation(params_a[0] + params_a[1] * x1 + params_a[2] * x2 + params_a[3] * x3)
    b_t = activation(params_b[0] + params_b[1] * x1 + params_b[2] * x2 + params_b[3] * x3)
    c_t = activation(params_c[0] + params_c[1] * x1 + params_c[2] * x2 + params_c[3] * x3)
    
    # Compute interaction terms
    inter_12 = params_inter[0] * x1 * x2
    inter_13 = params_inter[1] * x1 * x3
    inter_23 = params_inter[2] * x2 * x3
    
    # Compute log-odds and probabilities
    log_odds = a_t * x1 + b_t * x2 + c_t * x3 + inter_12 + inter_13 + inter_23
    p = 1 / (1 + np.exp(-log_odds))
    
    # Compute negative log-likelihood
    loss = -np.mean(y * np.log(p + 1e-10) + (1 - y) * np.log(1 - p + 1e-10))
    return loss

# Optimize the custom logistic regression model
def fit_custom_model_3_predictors(train_data, activation):
    initial_params = np.random.normal(0, 0.1, 15)  # Updated to 15 parameters (4 for each weight + 3 for interactions)
    result = minimize(
        custom_logistic_loss_3_predictors, 
        initial_params, 
        args=(train_data['x1'], train_data['x2'], train_data['x3'], train_data['y'], activation), 
        method='L-BFGS-B'
    )
    return result.x

# Compute KS statistic
def compute_ks(y_true, y_pred):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    ks = max(tpr - fpr)
    return ks

# Evaluate and compare models
def evaluate_models_3_predictors(train_data, test_data, custom_params, activation):
    X_train, y_train = train_data[['x1', 'x2', 'x3']], train_data['y']
    X_test, y_test = test_data[['x1', 'x2', 'x3']], test_data['y']

    results = {}

    # General logistic regression
    general_lr = LogisticRegression()
    general_lr.fit(X_train, y_train)
    general_p_test = general_lr.predict_proba(X_test)[:, 1]
    general_predictions_test = general_lr.predict(X_test)
    results['General Logistic Regression'] = {
        'Log-Loss': log_loss(y_test, general_p_test),
        'Accuracy': accuracy_score(y_test, general_predictions_test),
        'AUC': roc_auc_score(y_test, general_p_test),
        'KS': compute_ks(y_test, general_p_test)
    }

    # Custom logistic regression
    params_a = custom_params[:4]
    params_b = custom_params[4:8]
    params_c = custom_params[8:12]
    params_inter = custom_params[12:]
    custom_log_odds = (
        activation(params_a[0] + params_a[1] * X_test['x1'] + params_a[2] * X_test['x2'] + params_a[3] * X_test['x3']) * X_test['x1'] +
        activation(params_b[0] + params_b[1] * X_test['x1'] + params_b[2] * X_test['x2'] + params_b[3] * X_test['x3']) * X_test['x2'] +
        activation(params_c[0] + params_c[1] * X_test['x1'] + params_c[2] * X_test['x2'] + params_c[3] * X_test['x3']) * X_test['x3'] +
        params_inter[0] * X_test['x1'] * X_test['x2'] +
        params_inter[1] * X_test['x1'] * X_test['x3'] +
        params_inter[2] * X_test['x2'] * X_test['x3']
    )
    custom_p_test = 1 / (1 + np.exp(-custom_log_odds))
    custom_predictions_test = (custom_p_test > 0.5).astype(int)
    results['Custom Logistic Regression'] = {
        'Log-Loss': log_loss(y_test, custom_p_test),
        'Accuracy': accuracy_score(y_test, custom_predictions_test),
        'AUC': roc_auc_score(y_test, custom_p_test),
        'KS': compute_ks(y_test, custom_p_test)
    }

    # GAM model
    gam = LogisticGAM(s(0) + s(1) + s(2))
    gam.fit(X_train, y_train)
    gam_p_test = gam.predict_proba(X_test)
    gam_predictions_test = gam.predict(X_test)
    results['GAM'] = {
        'Log-Loss': log_loss(y_test, gam_p_test),
        'Accuracy': accuracy_score(y_test, gam_predictions_test),
        'AUC': roc_auc_score(y_test, gam_p_test),
        'KS': compute_ks(y_test, gam_p_test)
    }

    # XGBoost
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_p_test = xgb_model.predict_proba(X_test)[:, 1]
    xgb_predictions_test = xgb_model.predict(X_test)
    results['XGBoost'] = {
        'Log-Loss': log_loss(y_test, xgb_p_test),
        'Accuracy': accuracy_score(y_test, xgb_predictions_test),
        'AUC': roc_auc_score(y_test, xgb_p_test),
        'KS': compute_ks(y_test, xgb_p_test)
    }

    # Random Forest
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)
    rf_p_test = rf_model.predict_proba(X_test)[:, 1]
    rf_predictions_test = rf_model.predict(X_test)
    results['Random Forest'] = {
        'Log-Loss': log_loss(y_test, rf_p_test),
        'Accuracy': accuracy_score(y_test, rf_predictions_test),
        'AUC': roc_auc_score(y_test, rf_p_test),
        'KS': compute_ks(y_test, rf_p_test)
    }

    # LightGBM
    lgbm_model = LGBMClassifier(random_state=42)
    lgbm_model.fit(X_train, y_train)
    lgbm_p_test = lgbm_model.predict_proba(X_test)[:, 1]
    lgbm_predictions_test = lgbm_model.predict(X_test)
    results['LightGBM'] = {
        'Log-Loss': log_loss(y_test, lgbm_p_test),
        'Accuracy': accuracy_score(y_test, lgbm_predictions_test),
        'AUC': roc_auc_score(y_test, lgbm_p_test),
        'KS': compute_ks(y_test, lgbm_p_test)
    }

    return results

# Main Experiment
data = pd.read_csv('data_weight_func.csv')

# Split the data
train_data, test_data = train_test_split(data, test_size=0.5, random_state=42)

# Test different activation functions
activations = {'Sigmoid': sigmoid, 'ReLU': relu, 'Tanh': tanh, 'Softplus': softplus}

for name, activation in activations.items():
    print(f"\nUsing Activation Function: {name}")
    custom_params = fit_custom_model_3_predictors(train_data, activation)
    results = evaluate_models_3_predictors(train_data, test_data, custom_params, activation)
    for model, metrics in results.items():
        print(f"{model}: Log-Loss = {metrics['Log-Loss']:.4f}, Accuracy = {metrics['Accuracy']:.4f}, AUC = {metrics['AUC']:.4f}, KS = {metrics['KS']:.4f}")


###############DINN#################
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Define DINN model
class DINN(nn.Module):
    def __init__(self, input_dim, activation=torch.sigmoid):
        super(DINN, self).__init__()
        self.input_dim = input_dim
        self.activation = activation

        # Dynamic weights for each input feature
        self.dynamic_weights = nn.ModuleList([nn.Linear(input_dim, 1) for _ in range(input_dim)])

        # Pairwise interaction terms
        self.interaction_weights = nn.Parameter(torch.randn(int(input_dim * (input_dim - 1) / 2)))

    def forward(self, x):
        # Compute dynamic weights
        dynamic_weights = [self.activation(layer(x)) for layer in self.dynamic_weights]

        # Compute weighted features
        weighted_features = [dynamic_weights[i] * x[:, i].view(-1, 1) for i in range(self.input_dim)]

        # Compute interaction terms
        interaction_index = 0
        interaction_terms = []
        for i in range(self.input_dim):
            for j in range(i + 1, self.input_dim):
                interaction = self.interaction_weights[interaction_index] * x[:, i] * x[:, j]
                interaction_terms.append(interaction.view(-1, 1))
                interaction_index += 1

        # Combine features and interactions into log-odds
        log_odds = torch.sum(torch.cat(weighted_features, dim=1), dim=1, keepdim=True) + \
                   torch.sum(torch.cat(interaction_terms, dim=1), dim=1, keepdim=True)

        # Output probabilities
        probs = torch.sigmoid(log_odds)
        return probs

# Split dataset
data = pd.read_csv('data_weight_func.csv')
X = data[['x1', 'x2', 'x3']].values
y = data['y'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Model, loss, and optimizer
input_dim = X_train.shape[1]
model = DINN(input_dim=input_dim, activation=torch.sigmoid)
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 100
for epoch in range(epochs):
    # Forward pass
    probs = model(X_train)
    loss = criterion(probs, y_train)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Logging
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Evaluate performance
with torch.no_grad():
    y_pred_proba = model(X_test).numpy()
    y_pred = (y_pred_proba > 0.5).astype(int)
    logloss = log_loss(y_test.numpy(), y_pred_proba)
    accuracy = accuracy_score(y_test.numpy(), y_pred)
    auc = roc_auc_score(y_test.numpy(), y_pred_proba)

    print("\nEvaluation Metrics:")
    print(f"Log-Loss: {logloss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")