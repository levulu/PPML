import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Load Helpdesk dataset
data_path = 'helpdesk.csv'
helpdesk_data = pd.read_csv(data_path)

# Preprocessing
helpdesk_data['CompleteTimestamp'] = pd.to_datetime(helpdesk_data['CompleteTimestamp'])
helpdesk_data = helpdesk_data.sort_values(by=['CaseID', 'CompleteTimestamp'])
helpdesk_data['ActivityID'] = helpdesk_data['ActivityID'].astype('category')
helpdesk_data['ActivityCode'] = helpdesk_data['ActivityID'].cat.codes

# Create sequences and targets
grouped = helpdesk_data.groupby('CaseID')
sequences = []
targets = []
for _, group in grouped:
    events = group['ActivityCode'].tolist()
    for i in range(1, len(events)):
        sequences.append(events[:i])
        targets.append(events[i])

# Parameters
num_activities = helpdesk_data['ActivityCode'].nunique()
max_sequence_length = max(len(seq) for seq in sequences)
embedding_dim = 50
lstm_units = 64

# Pad sequences and encode targets
X = pad_sequences(sequences, maxlen=max_sequence_length, padding='pre')
y = to_categorical(targets, num_classes=num_activities)

# Build the LSTM model
model = Sequential([
    Masking(mask_value=0, input_shape=(max_sequence_length,)),
    Embedding(input_dim=num_activities, output_dim=embedding_dim, input_length=max_sequence_length),
    LSTM(units=lstm_units, return_sequences=False),
    Dense(units=num_activities, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2, verbose=1)

# Save the model
model.save('helpdesk_lstm_model.h5')

# Print final training and validation accuracy
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
print(f"Final Training Accuracy: {final_train_acc}")
print(f"Final Validation Accuracy: {final_val_acc}")