import time
import numpy as np
from keras.models import Sequential
from keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import model_from_json

plain16 = np.load("FinalPlain32.npy")
cipher16 = np.load("FinalTrain32.npy")
print("Data is loaded!")

X = cipher16
y = plain16

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2)
print("Data splitted!")

model = Sequential()
model.add(layers.Dense(128, input_dim=128, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(128, activation='sigmoid'))
print("Model set!")

start = time.perf_counter()
model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
print(model.summary())
end = time.perf_counter()
print('Time for model.compile ', (end-start)/60, "min")

start = time.perf_counter()
# history = model.fit(X_train, y_train, epochs=100, verbose=1,  validation_split=0.2, batch_size=5000)
history = model.fit(X_train, y_train, epochs=100, verbose=2,  validation_data=(X_test,y_test), batch_size=5000)
#validation_data=(X_test,y_test)
end = time.perf_counter()
print('Time for model.fit ', (end-start)/60, 'min')

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

eval = model.evaluate(X_test, y_test)
print(eval)

model_json = model.to_json()
with open("model32.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model32.h5")
print("Saved model to disk")
