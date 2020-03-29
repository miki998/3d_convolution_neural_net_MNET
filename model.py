import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import h5py


def array_to_color(array, cmap="Oranges"):
	s_m = plt.cm.ScalarMappable(cmap=cmap)
	return s_m.to_rgba(array)[:,:-1]


def rgb_data_transform(data):
	data_t = []
	for i in range(data.shape[0]):
		data_t.append(array_to_color(data[i]).reshape(16, 16, 16, 3))
	return np.asarray(data_t, dtype=np.float32)


with h5py.File("./full_dataset_vectors.h5", "r") as hf:    

	# Split the data into training/test features/targets
	X_train = hf["X_train"][:]
	targets_train = hf["y_train"][:]
	X_test = hf["X_test"][:] 
	targets_test = hf["y_test"][:]

	# Determine sample shape
	sample_shape = (16, 16, 16, 3)

	# Reshape data into 3D format
	X_train = rgb_data_transform(X_train)
	X_test = rgb_data_transform(X_test)

	# Convert target vectors to categorical targets
	targets_train = to_categorical(targets_train).astype(np.integer)
	targets_test = to_categorical(targets_test).astype(np.integer)
	
	# Create the model
	model = Sequential()
	model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=sample_shape))
	model.add(MaxPooling3D(pool_size=(2, 2, 2)))
	model.add(Dropout(0.5))
	model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', kernel_initializer='he_uniform'))
	model.add(MaxPooling3D(pool_size=(2, 2, 2)))
	model.add(Dropout(0.5))
	model.add(Flatten())
	model.add(Dense(256, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))

	# Compile the model
	model.compile(loss='categorical_crossentropy',
				  optimizer=keras.optimizers.Adam(lr=0.001),
				  metrics=['accuracy'])

	# Fit data to model
	history = model.fit(X_train, targets_train,
				batch_size=128,
				epochs=30,
				verbose=1,
				validation_split=0.2)

	# Generate generalization metrics
	score = model.evaluate(X_test, targets_test, verbose=0)
	print(f'Test loss: {score[0]} / Test accuracy: {score[1]}')
	model.save('threeD.h5')
	print(history.history.keys())
	with open('history.pickle','wb') as handle:
		pickle.dump(history.history, handle)

	# Plot history: Categorical crossentropy & Accuracy
	# plt.plot(history.history['loss'], label='Categorical crossentropy (training data)')
	# plt.plot(history.history['val_loss'], label='Categorical crossentropy (validation data)')
	# plt.plot(history.history['accuracy'], label='Accuracy (training data)')
	# plt.plot(history.history['val_accuracy'], label='Accuracy (validation data)')
	# plt.title('Model performance for 3D MNIST Keras Conv3D example')
	# plt.ylabel('Loss value')
	# plt.xlabel('No. epoch')
	# plt.legend(loc="upper left")
	# plt.show()