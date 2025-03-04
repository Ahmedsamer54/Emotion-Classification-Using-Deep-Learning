"""
PyCharm 2024.1.4 (Community Edition)
Build #PC-241.18034.82, built on June 24, 2024
Runtime version: 17.0.11+1-b1207.24 amd64
VM: OpenJDK 64-Bit Server VM by JetBrains s.r.o.
Windows 11.0
GC: G1 Young Generation, G1 Old Generation
Memory: 2048M
Cores: 12
Registry:
  ide.experimental.ui=true
Non-Bundled Plugins:
  net.seesharpsoft.intellij.plugins.csv (3.3.0-241)
"""

import torch
import torch.nn as nn
import torch.nn.functional as f
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split as t
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np


# Function 1: general function to generate heartrate data
def generate_heart_rate_data(num_patients, num_seconds, bpm_range, bias=None):
    if bias:
        data = np.random.normal(loc=bias, scale=(bpm_range[1] - bpm_range[0]) / 6, size=(num_patients, num_seconds))
    else:
        data = np.random.uniform(low=bpm_range[0], high=bpm_range[1], size=(num_patients, num_seconds))
    return np.clip(data, bpm_range[0], bpm_range[1])


# Function 2: uses function 1 to generate heartrate data and adds mood labels
def generate_mood_heart_rate_data(num_patients, duration):
    # Define the mood ranges (in BPM)
    mood_ranges = {
        'neutral': (65, 75),
        'happy': (60, 65),
        'sad': (75, 100),
        'angry': (80, 110)
    }

    # Generate data for each mood
    for mood, bpm_range in mood_ranges.items():
        if mood == 'neutral':
            data = generate_heart_rate_data(num_patients, duration, bpm_range, bias=70)
        else:
            data = generate_heart_rate_data(num_patients, duration, bpm_range)

        # Create a DataFrame
        df = pd.DataFrame(data)

        # Save to CSV
        df.to_csv(f'{mood}_heart_rate.csv', index=False, header=False)


# Function 3: generates SpO2 data with mood labels
def generate_spo2_data(num_patients, duration):
    # Define the mood ranges
    mood_ranges = {
        'neutral': (99, 100),
        'happy': (98, 99),
        'sad': (85, 100),
        'angry': (98, 100)
    }

    # Create a dictionary to hold DataFrames for each mood
    mood_dataframes = {}

    # Generate data for each mood
    for mood, (low, high) in mood_ranges.items():
        # Generate random SpO2 data
        data = np.random.uniform(low, high, (num_patients, duration))
        # Create a DataFrame
        mood_dataframes[mood] = pd.DataFrame(data)
        # Save to CSV
        mood_dataframes[mood].to_csv(f'{mood}_spo2.csv', index=False, header=False)


# Function 4: generates body temperature data with mood labels
def generate_body_temperature_data(num_patients, duration):
    # Define the mood ranges (in °C)
    mood_ranges = {
        'neutral': (36.5, 37.5),
        'happy': (36.0, 37.0),
        'sad': (36.5, 38.0),
        'angry': (37.0, 38.5)
    }

    # Generate data for each mood
    for mood, temp_range in mood_ranges.items():
        data = np.random.uniform(low=temp_range[0], high=temp_range[1], size=(num_patients, duration))
        # Create a DataFrame
        df = pd.DataFrame(data)
        # Save to CSV
        df.to_csv(f'{mood}_body_temperature.csv', index=False, header=False)


# Parameters for generating patient data
patients = 1000  # Number of patients
seconds = 120  # Number of seconds
generate_spo2_data(patients, seconds)
generate_mood_heart_rate_data(patients, seconds)
generate_body_temperature_data(patients, seconds)


# Defining neural network class with dropout layers
class Model(nn.Module):
    def __init__(self, input_f, h1=256, h2=128, h3=64, h4=32, output_f=4):
        super().__init__()
        self.fc1 = nn.Linear(input_f, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3, h4)
        self.out = nn.Linear(h4, output_f)
        self.dropout = nn.Dropout(0.2)
        self.batch_norm1 = nn.BatchNorm1d(h1)
        self.batch_norm2 = nn.BatchNorm1d(h2)
        self.batch_norm3 = nn.BatchNorm1d(h3)
        self.batch_norm4 = nn.BatchNorm1d(h4)

    def forward(self, element):
        element = self.dropout(f.relu(self.batch_norm1(self.fc1(element))))
        element = self.dropout(f.relu(self.batch_norm2(self.fc2(element))))
        element = self.dropout(f.relu(self.batch_norm3(self.fc3(element))))
        element = self.dropout(f.relu(self.batch_norm4(self.fc4(element))))
        element = self.out(element)
        return element


# Function to calculate heart rate features
def calculate_hr_features(heart_rate_data):
    hr_data = np.array(heart_rate_data)
    mean_hr = np.mean(hr_data)
    std_hr = np.std(hr_data)
    min_hr = np.min(hr_data)
    max_hr = np.max(hr_data)
    rmssd = np.sqrt(np.mean(np.diff(hr_data) ** 2))  # Root-mean-square of successive differences

    return {
        'mean_hr': mean_hr,
        'std_hr': std_hr,
        'min_hr': min_hr,
        'max_hr': max_hr,
        'rmssd': rmssd
    }


# Function to calculate SpO2 features
def calculate_spo2_features(spo2_data):
    spo2_data = np.array(spo2_data)
    mean_spo2 = np.mean(spo2_data)
    std_spo2 = np.std(spo2_data)
    min_spo2 = np.min(spo2_data)
    max_spo2 = np.max(spo2_data)
    range_spo2 = max_spo2 - min_spo2  # Range of SpO2 values

    return {
        'mean_spo2': mean_spo2,
        'std_spo2': std_spo2,
        'min_spo2': min_spo2,
        'max_spo2': max_spo2,
        'range_spo2': range_spo2
    }


# Function to calculate body temperature features
def calculate_temperature_features(temp_data):
    temp_data = np.array(temp_data)
    mean_temp = np.mean(temp_data)
    std_temp = np.std(temp_data)
    min_temp = np.min(temp_data)
    max_temp = np.max(temp_data)
    range_temp = max_temp - min_temp  # Range of body temperature values

    return {
        'mean_temp': mean_temp,
        'std_temp': std_temp,
        'min_temp': min_temp,
        'max_temp': max_temp,
        'range_temp': range_temp
    }


# Reading the CSV files for heart rate data
angry_hr_df = pd.read_csv('angry_heart_rate.csv', header=None)
neutral_hr_df = pd.read_csv('neutral_heart_rate.csv', header=None)
sad_hr_df = pd.read_csv('sad_heart_rate.csv', header=None)
happy_hr_df = pd.read_csv('happy_heart_rate.csv', header=None)

# Reading the CSV files for SpO2 data
angry_spo2_df = pd.read_csv('angry_spo2.csv', header=None)
neutral_spo2_df = pd.read_csv('neutral_spo2.csv', header=None)
sad_spo2_df = pd.read_csv('sad_spo2.csv', header=None)
happy_spo2_df = pd.read_csv('happy_spo2.csv', header=None)

# Reading the CSV files for body temperature data
angry_temperature_df = pd.read_csv('angry_body_temperature.csv', header=None)
neutral_temperature_df = pd.read_csv('neutral_body_temperature.csv', header=None)
sad_temperature_df = pd.read_csv('sad_body_temperature.csv', header=None)
happy_temperature_df = pd.read_csv('happy_body_temperature.csv', header=None)

# Adding mood labels to heart rate data
neutral_hr_df['mood'] = 0
happy_hr_df['mood'] = 1
sad_hr_df['mood'] = 2
angry_hr_df['mood'] = 3

# Adding mood labels to SpO2 data
neutral_spo2_df['mood'] = 0
happy_spo2_df['mood'] = 1
sad_spo2_df['mood'] = 2
angry_spo2_df['mood'] = 3

# Adding mood labels to body temperature data
neutral_temperature_df['mood'] = 0
happy_temperature_df['mood'] = 1
sad_temperature_df['mood'] = 2
angry_temperature_df['mood'] = 3

# Calculate mean neutral heart rate and SpO2
mean_neutral_hr = neutral_hr_df.drop('mood', axis=1).mean().values
mean_neutral_spo2 = neutral_spo2_df.drop('mood', axis=1).mean().values
mean_neutral_temperature = neutral_temperature_df.drop('mood', axis=1).mean().values

# Subtract mean neutral heart rate from all heart rate dataframes
angry_hr_df.iloc[:, :-1] -= mean_neutral_hr
neutral_hr_df.iloc[:, :-1] -= mean_neutral_hr
sad_hr_df.iloc[:, :-1] -= mean_neutral_hr
happy_hr_df.iloc[:, :-1] -= mean_neutral_hr

# Subtract mean neutral SpO2 from all SpO2 dataframes
angry_spo2_df.iloc[:, :-1] -= mean_neutral_spo2
neutral_spo2_df.iloc[:, :-1] -= mean_neutral_spo2
sad_spo2_df.iloc[:, :-1] -= mean_neutral_spo2
happy_spo2_df.iloc[:, :-1] -= mean_neutral_spo2

# Subtract mean neutral body temperature from all body temperature dataframes
angry_temperature_df.iloc[:, :-1] -= mean_neutral_temperature
neutral_temperature_df.iloc[:, :-1] -= mean_neutral_temperature
sad_temperature_df.iloc[:, :-1] -= mean_neutral_temperature
happy_temperature_df.iloc[:, :-1] -= mean_neutral_temperature

# Combining all heart rate datasets
combined_hr_df = pd.concat([angry_hr_df, neutral_hr_df, sad_hr_df, happy_hr_df], ignore_index=True)

# Combining all SpO2 datasets
combined_spo2_df = pd.concat([angry_spo2_df, neutral_spo2_df, sad_spo2_df, happy_spo2_df], ignore_index=True)

# Combining all body temperature datasets
combined_temperature_df = pd.concat(
    [angry_temperature_df, neutral_temperature_df, sad_temperature_df, happy_temperature_df], ignore_index=True)

# Create new DataFrame with features
combined_features = []
for index, row in combined_hr_df.drop('mood', axis=1).iterrows():
    hr_features = calculate_hr_features(row.values)
    spo2_features = calculate_spo2_features(combined_spo2_df.loc[index].drop('mood').values)
    temperature_features = calculate_temperature_features(combined_temperature_df.loc[index].drop('mood').values)

    # Combine features
    combined_feature = {**hr_features, **spo2_features, **temperature_features,
                        'mood': combined_hr_df.loc[index, 'mood']}
    combined_features.append(combined_feature)

features_df = pd.DataFrame(combined_features)

# Save combined features to CSV
features_df.to_csv('combined_features.csv', index=False)

# Splitting data into input (X) and output (y)
X = features_df.drop('mood', axis=1)
y = features_df['mood']

# Converting data to arrays
X = X.values
y = y.values

# Splitting dataset
X_train, X_test, y_train, y_test = t(X, y, test_size=0.25, random_state=32)

# Converting to tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)

# Creating instance of nn
input_features = X_train.shape[1]
model = Model(input_features)

# Setting criterion and optimizer
crit = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.001)

# Learning rate scheduler
scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10, verbose=True)

# Number of epochs
epochs = 2000

# Lists to track metrics
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Early stopping parameters
patience = 20  # Number of epochs to wait for improvement
best_val_loss = float('inf')  # Initialize best validation loss
epochs_since_improvement = 0  # Counter for epochs since last improvement

# Training neural network with validation
for i in range(epochs):
    # Training phase
    model.train()
    y_pred = model.forward(X_train)
    train_loss = crit(y_pred, y_train)
    train_losses.append(train_loss.item())

    # Calculate training accuracy
    _, predicted = torch.max(y_pred.data, 1)
    train_acc = (predicted == y_train).sum().item() / len(y_train)
    train_accuracies.append(train_acc)

    # Backpropagation
    opt.zero_grad()
    train_loss.backward()
    opt.step()

    # Validation phase
    model.eval()
    with torch.no_grad():
        y_val = model.forward(X_test)
        val_loss = crit(y_val, y_test)
        val_losses.append(val_loss.item())

        # Calculate validation accuracy
        _, predicted = torch.max(y_val.data, 1)
        val_acc = (predicted == y_test).sum().item() / len(y_test)
        val_accuracies.append(val_acc)

    # Update learning rate
    scheduler.step(val_loss)

    # Check for improvement in validation loss
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_since_improvement = 0  # Reset counter if improvement is found
    else:
        epochs_since_improvement += 1  # Increment counter if no improvement

    if i % 10 == 0:
        print(f'Epoch: {i}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
              f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')

    # Check for early stopping
    if epochs_since_improvement >= patience:
        print(f'Early stopping triggered after {i + 1} epochs without improvement.')
        break

# Individual prediction evaluation
print("\n=== Individual Prediction Evaluation ===")
correct = 0
with torch.no_grad():
    y_eval = model.forward(X_test)
    loss = crit(y_eval, y_test)
    print(f'Test Loss: {loss}')

    for i, data in enumerate(X_test):
        # Reshape the data to have batch dimension and feature dimension
        data = data.view(1, -1)
        y_val = model.forward(data)
        predicted = y_val.argmax().item()
        actual = y_test[i]
        if predicted == actual:
            correct += 1
        print(
            f'Sample {i + 1}: Predicted: {predicted} ({"Neutral" if predicted == 0 else "Happy" if predicted == 1 else "Sad" if predicted == 2 else "Angry"}), '
            f'Actual: {actual} ({"Neutral" if actual == 0 else "Happy" if actual == 1 else "Sad" if actual == 2 else "Angry"})')

print(f'\nAccuracy: {correct / len(y_test) * 100:.2f}%')

# Final Validation and Testing
print("\n=== Final Model Evaluation ===")

# Set model to evaluation mode
model.eval()
with torch.no_grad():
    # Final validation metrics
    final_val_pred = model.forward(X_test)
    final_val_loss = crit(final_val_pred, y_test)
    _, predicted_val = torch.max(final_val_pred.data, 1)
    final_val_accuracy = (predicted_val == y_test).sum().item() / len(y_test)

    # Calculate validation metrics per class
    val_confusion = torch.zeros(4, 4)
    for i in range(len(y_test)):
        val_confusion[y_test[i], predicted_val[i]] += 1

    # Calculate precision, recall, and F1 score for each class
    val_precision = torch.zeros(4)
    val_recall = torch.zeros(4)
    val_f1 = torch.zeros(4)

    for i in range(4):
        true_positive = val_confusion[i, i]
        false_positive = val_confusion[:, i].sum() - true_positive
        false_negative = val_confusion[i, :].sum() - true_positive

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        val_precision[i] = precision
        val_recall[i] = recall
        val_f1[i] = f1

# Print final results
print(f"\nFinal Validation Loss: {final_val_loss:.4f}")
print(f"Final Validation Accuracy: {final_val_accuracy:.4f}")

# Print metrics for each class
emotion_labels = ['Neutral', 'Happy', 'Sad', 'Angry']
print("\nPer-class Metrics:")
print("Class\t\tPrecision\tRecall\t\tF1-Score")
print("-" * 60)
for i in range(4):
    print(f"{emotion_labels[i]:<12}\t{val_precision[i]:.4f}\t\t{val_recall[i]:.4f}\t\t{val_f1[i]:.4f}")

# Print confusion matrix
print("\nConfusion Matrix:")
print("\t\t" + "\t".join(emotion_labels))
for i in range(4):
    print(f"{emotion_labels[i]:<8}\t", end="")
    for j in range(4):
        print(f"{int(val_confusion[i, j])}\t\t", end="")
    print()

# Save model
torch.save(model.state_dict(), 'emotion_classifier_model.pth')

# Plot confusion matrix
plt.figure(figsize=(10, 8))
plt.imshow(val_confusion.numpy(), cmap='coolwarm')
plt.colorbar()
plt.xticks(range(4), emotion_labels, rotation=45)
plt.yticks(range(4), emotion_labels)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')

# Add numbers to the plot
for i in range(4):
    for j in range(4):
        plt.text(j, i, int(val_confusion[i, j]),
                 ha='center', va='center')

plt.tight_layout()
plt.show()

# Plot ROC curves
plt.figure(figsize=(10, 8))
for i in range(4):
    # Calculate ROC curve for each class
    y_true_binary = (y_test == i).numpy()
    y_score = f.softmax(final_val_pred, dim=1)[:, i].numpy()

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(y_true_binary, y_score)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'{emotion_labels[i]} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend(loc="lower right")
plt.show()

# Learning curves
plt.figure(figsize=(15, 5))

# Loss curves
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Accuracy curves
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Save final metrics to a file
with open('model_evaluation_results.txt', 'w') as f:
    f.write("=== Model Evaluation Results ===\n\n")
    f.write(f"Final Validation Loss: {final_val_loss:.4f}\n")
    f.write(f"Final Validation Accuracy: {final_val_accuracy:.4f}\n\n")

    f.write("Per-class Metrics:\n")
    f.write("Class\t\tPrecision\tRecall\t\tF1-Score\n")
    f.write("-" * 60 + "\n")
    for i in range(4):
        f.write(f"{emotion_labels[i]:<12}\t{val_precision[i]:.4f}\t\t{val_recall[i]:.4f}\t\t{val_f1[i]:.4f}\n")
