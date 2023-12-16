import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Conv2D, MaxPooling2D, Flatten, BatchNormalization,Dropout
import matplotlib.pyplot as plt

print("---------------------------------------------------------")
print("GPU = ",len(tf.config.list_physical_devices('GPU')))
print("---------------------------------------------------------")
# generators

train_ds = keras.utils.image_dataset_from_directory(
    directory = 'dataset/train',
    labels = 'inferred',
    label_mode = 'int',
    batch_size=32,
    image_size=(256,256)
)

validation_ds = keras.utils.image_dataset_from_directory(
    directory = 'dataset/test',
    labels = 'inferred',
    label_mode = 'int',
    batch_size=32,
    image_size=(256,256)
)

# Normalization
def process(image,label):
    image = tf.cast(image/255, tf.float32)
    return image,label

train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)

# CNN model
model = Sequential()
model.add(Conv2D(32,kernel_size = (3,3), padding ='valid', activation = 'relu', input_shape=(256,256,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding ='valid'))

model.add(Conv2D(64,kernel_size = (3,3), padding ='valid', activation = 'relu', input_shape=(256,256,3)))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding ='valid'))

model.add(Conv2D(128,kernel_size = (3,3), padding ='valid', activation = 'relu', input_shape=(256,256,3)))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding ='valid'))

model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer ='adam', loss='binary_crossentropy', metrics =['accuracy'])
history = model.fit(train_ds, epochs=10, validation_data=validation_ds)

#graphs
plt.plot(history.history['accuracy'],color='red',label='train')
plt.plot(history.history['val_accuracy'],color='blue',label='train')
plt.legend()
plt.show()

plt.plot(history.history['loss'],color='red',label='train')
plt.plot(history.history['val_loss'],color='blue',label='train')
plt.legend()
plt.show()
