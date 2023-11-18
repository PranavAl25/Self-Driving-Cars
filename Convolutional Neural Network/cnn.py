cnn = Sequential()
cnn.add(Conv2D(64, (3, 3), input_shape=(32, 32, 3)))
cnn.add(Activation('relu'))
cnn.add(MaxPooling2D(pool_size=(2, 2)))
cnn.add(Flatten())
cnn.add(Dense(units = 128, activation = 'relu'))
cnn.add(Dense(units = 3, activation = 'softmax'))
cnn.compile(loss ='categorical_crossentropy', optimizer = optimizers.SGD(learning_rate=1e-3, momentum=0.95), metrics = ['accuracy'])

cnn.fit(X_train_norm,y_train_onehot, epochs=20, validation_data=(X_test_norm, y_test_onehot), shuffle = True, callbacks =[monitor])

plot_acc(cnn.history)

print(sliding_predictions(cnn, windows_numpy, 0.6))
