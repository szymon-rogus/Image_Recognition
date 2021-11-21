from numpy import load
import matplotlib.pyplot as plt

history = load('history8.npy', allow_pickle=True).item()

plt.plot(history['accuracy'], label='accuracy')
plt.plot(history['val_accuracy'], label='val_accuracy')
plt.plot(history['loss'], label='loss')
plt.plot(history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 2.5])
plt.legend(loc='lower right')
plt.show()
