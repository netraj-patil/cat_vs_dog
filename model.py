import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense,Conv2D, MaxPooling2D, Flatten, BatchNormalization,Dropout

# generators

train_ds = keras.utils.image_dataset_from_directory(
    directory = 'datasets/train',
    labels = 'inferred',
    label_mode = 'int',
    batch_size=10,
    image_size=(256,256)
)

validation_ds = keras.utils.image_dataset_from_directory(
    directory = 'datasets/test',
    labels = 'inferred',
    label_mode = 'int',
    batch_size=10,
    image_size=(256,256)
)

# Normalization of data
def process(image,label):
    image = tf.cast(image/255, tf.float32)
    return image,label

train_ds = train_ds.map(process)
validation_ds = validation_ds.map(process)

# CNN model
model = Sequential()
model.add(Conv2D(32,kernel_size = (3,3), padding ='valid', activation = 'relu', input_shape=(256,256,3)))
model.add(Conv2D(64,kernel_size = (3,3), padding ='valid', activation = 'relu', input_shape=(256,256,3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding ='valid'))

model.add(Conv2D(128,kernel_size = (3,3), padding ='valid', activation = 'relu', input_shape=(256,256,3)))
model.add(Conv2D(128,kernel_size = (3,3), padding ='valid', activation = 'relu', input_shape=(256,256,3)))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding ='valid'))

model.add(Conv2D(256,kernel_size = (3,3), padding ='valid', activation = 'relu', input_shape=(256,256,3)))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding ='valid'))

model.add(Conv2D(512,kernel_size = (3,3), padding ='valid', activation = 'relu', input_shape=(256,256,3)))
model.add(Dropout(0.1))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding ='valid'))

model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64,activation='relu'))
model.add(Dense(1,activation='sigmoid'))


model.compile(optimizer ='adam', loss='binary_crossentropy', metrics =['accuracy'])

# here we have used Early stopping to reduce overfitting
callback = EarlyStopping(
    monitor="val_loss",
    min_delta=0.0001,
    patience=3,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
)

history = model.fit(train_ds, epochs=10, validation_data=validation_ds, callbacks=callback)

# save the model
model.save('model2.h5')