import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Masking, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanAbsoluteError
import matplotlib.pyplot as plt

# Load Helpdesk dataset 
data_path = 'helpdesk.csv'  
helpdesk_data = pd.read_csv(data_path)

# Preprocessing
helpdesk_data['CompleteTimestamp'] = pd.to_datetime(helpdesk_data['CompleteTimestamp'])
helpdesk_data = helpdesk_data.sort_values(by=['CaseID', 'CompleteTimestamp'])
helpdesk_data['ActivityID'] = helpdesk_data['ActivityID'].astype('category')
helpdesk_data['ActivityCode'] = helpdesk_data['ActivityID'].cat.codes

# Create sequences and remaining time targets
grouped = helpdesk_data.groupby('CaseID')
sequences = []
remaining_times = []
for _, group in grouped:
    events = group['ActivityCode'].tolist()
    timestamps = group['CompleteTimestamp'].tolist()
    for i in range(1, len(events)):
        sequences.append(events[:i])
        remaining_time = (timestamps[-1] - timestamps[i - 1]).total_seconds() / 3600  # Remaining time in hours
        remaining_times.append(remaining_time)

# Pad sequences
max_sequence_length = max(len(seq) for seq in sequences)
X = pad_sequences(sequences, maxlen=max_sequence_length, padding='pre')
y_remaining_time = np.array(remaining_times)

# Define the LSTM model structure for regression
def build_model(input_length, num_activities, embedding_dim=50, lstm_units=64):
    model = Sequential([
        Input(shape=(input_length,)),
        Embedding(input_dim=num_activities, output_dim=embedding_dim, mask_zero=True),
        LSTM(units=lstm_units, return_sequences=False),
        Dense(units=1, activation='linear')  # Linear activation for continuous output (remaining time)
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanAbsoluteError(), metrics=['mae'])
    return model

# Perform K-fold cross-validation
k = 5  # Number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)
fold_no = 1
mae_scores = []

for train_index, val_index in kf.split(X):
    print(f"Fold {fold_no} training...")
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y_remaining_time[train_index], y_remaining_time[val_index]
    
    # Build model for each fold
    model = build_model(max_sequence_length, helpdesk_data['ActivityCode'].nunique())
    
    # Train model
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), verbose=1)
    
    # Evaluate model
    val_mae = model.evaluate(X_val, y_val, verbose=0)[1]  # Get MAE from metrics
    mae_scores.append(val_mae)
    
    print(f"Fold {fold_no} MAE: {val_mae}")
    fold_no += 1

# Print cross-validation results
mean_mae = np.mean(mae_scores)
print(f"\nAverage MAE across {k}-folds: {mean_mae:.4f}")

# Save final model
model.save('helpdesk_remaining_time_model.keras')

# Visual Analysis
# Plot MAE across folds
plt.figure(figsize=(8, 5))
plt.bar([f'Fold {i+1}' for i in range(k)], mae_scores, color='skyblue')
plt.xlabel('Fold')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('MAE Across K-Folds')
plt.show()

# Loss Progression for Last Fold
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), history.history['loss'], label='Training Loss')
plt.plot(range(1, 11), history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Absolute Error (MAE)')
plt.title('Training and Validation Loss Progression (Last Fold)')
plt.legend()
plt.grid(True)
plt.show()
