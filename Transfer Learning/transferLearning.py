vgg_expert = VGG16(weights = 'imagenet', include_top = False, input_shape = (32, 32, 3))

vgg_model = Sequential()
vgg_model.add(vgg_expert)

vgg_model = Sequential()
vgg_model.add(vgg_expert)
vgg_model.add(GlobalAveragePooling2D())
vgg_model.add(Dense(1024, activation = 'relu'))
vgg_model.add(Dropout(0.3))
vgg_model.add(Dense(3, activation = 'softmax'))

vgg_model.compile(loss ='categorical_crossentropy', optimizer = optimizers.SGD(learning_rate=1e-3, momentum=0.95), metrics = ['accuracy'])

vgg_model.fit(X_train_norm,y_train_onehot, epochs=20, validation_data=(X_test_norm, y_test_onehot), shuffle = True, callbacks =[monitor])

plot_acc(vgg_model.history)

acc = sliding_predictions(vgg_model, windows, threshold=0.9)
print("The accuracy is {}".format(acc))
