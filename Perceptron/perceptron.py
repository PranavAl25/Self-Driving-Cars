import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint

perceptron = Sequential()

perceptron.add(Flatten(input_shape = (32, 32, 3)))
perceptron.add(Dense(128, activation = 'relu'))
perceptron.add(Dense(3, activation = 'softmax'))
perceptron.compile(loss = 'categorical_crossentropy',
optimizer = optimizers.SGD(learning_rate=1e-3, momentum = 0.9),metrics = ['accuracy'])

# define our monitor. Don't worry about the parameters here except for './model.h5' which is the file that our model saves to
monitor = ModelCheckpoint('./model.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')

# Normalize the data.
X_train_norm = normalize(X_train)
X_test_norm = normalize(X_test)

# Convert labels into one-hot numpy arrays.
y_train_onehot = label_to_onehot(y_train)
y_test_onehot = label_to_onehot(y_test)

history = perceptron.fit(X_train_norm, y_train_onehot, epochs = 10, validation_data = (X_test_norm, y_test_onehot), shuffle = True, callbacks = [monitor])
