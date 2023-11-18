import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint

(X_train, y_train), (X_test, y_test) = load_vehicle_dataset()
perceptron = Sequential()
perceptron.add(Flatten(input_shape = (32, 32, 3)))
perceptron.add(Dense(units = 128, activation = 'relu'))
perceptron.add(Dense(units = 3, activation = 'softmax'))

perceptron.compile(loss='categorical_crossentropy',
                   optimizer=optimizers.SGD(learning_rate=1e-3, momentum=0.9),
                   metrics=['accuracy'])

monitor = ModelCheckpoint('./model.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')

X_train_norm = normalize(X_train)
X_test_norm = normalize(X_test)

y_train_onehot = label_to_onehot(y_train)
y_test_onehot = label_to_onehot(y_test)

# Train the model
history = perceptron.fit(X_train_norm, y_train_onehot, epochs=20, validation_data=(X_test_norm, y_test_onehot), shuffle=True, callbacks=[monitor])

plot_acc(history)

windows_norm = normalize(windows_numpy)
windows_pred = perceptron.predict(windows_norm)
y_pred = np.argmax(windows_pred, axis = -1)
pred_prob = np.max(windows_pred, axis = -1)
print(y_pred, pred_prob)

threshold = 0.6

num_windows = windows_numpy.shape[0]
for i in range(num_windows):
  if pred_prob[i] >= threshold and y_pred[i] > 0:
    plot_one_image(windows[i], labels=[" ".join([str(y_pred[i]), str(pred_prob[i])])], fig_size=(1,1))

np.mean(y_pred == labels)

def sliding_predictions(model, windows, threshold=0.6, labels=labels):
  windows_norm = normalize(windows_numpy)
  windows_pred = perceptron.predict(windows_norm)
  y_pred = np.argmax(windows_pred, axis = -1)
  pred_prob = np.max(windows_pred, axis = -1)
  print(y_pred, pred_prob)

  num_windows = windows_numpy.shape[0]

  for i in range(num_windows):
    if pred_prob[i] >= threshold and y_pred[i] > 0:
      plot_one_image(windows[i], labels=[" ".join([str(y_pred[i]), str(pred_prob[i])])], fig_size=(1,1))

  return np.mean(y_pred == labels)
