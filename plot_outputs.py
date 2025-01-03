import matplotlib.pyplot as plt

# Simulated data for training and validation accuracy/loss
epochs = range(1, 11)
training_accuracy = [65.81, 76.12, 76.87, 77.42, 76.55, 77.24, 77.47, 77.06, 76.84, 77.70]
validation_accuracy = [75.88, 77.19, 77.14, 77.19, 77.19, 77.19, 76.69, 77.19, 77.19, 77.19]
training_loss = [1.1014, 0.6752, 0.6770, 0.6576, 0.6670, 0.6427, 0.6603, 0.6437, 0.6481, 0.6322]
validation_loss = [0.6920, 0.6731, 0.6691, 0.6653, 0.6666, 0.6562, 0.6643, 0.6581, 0.6562, 0.6536]

# Plot training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(epochs, training_accuracy, label='Training Accuracy')
plt.plot(epochs, validation_accuracy, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('training_validation_accuracy.png')  # Save the figure

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(epochs, training_loss, label='Training Loss')
plt.plot(epochs, validation_loss, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_validation_loss.png')  # Save the figure

print("Graphs saved as 'training_validation_accuracy.png' and 'training_validation_loss.png'")