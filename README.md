# Dogs-vs.-Cats-Classification
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# 1. Data Preparation
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'path/to/training/images',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

test_set = test_datagen.flow_from_directory(
    'path/to/test/images',
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary')

# 2. Model Building
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# 3. Model Compilation
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Model Training
model.fit(
    x=training_set,
    validation_data=test_set,
    epochs=25
)

# 5. Model Evaluation
loss, accuracy = model.evaluate(test_set)
print('Accuracy:', accuracy)

# 6. Make Predictions
new_image = tf.keras.preprocessing.image.load_img('path/to/image.jpg', target_size=(64, 64))
new_image = tf.keras.preprocessing.image.img_to_array(new_image)
new_image = new_image / 255.0
new_image = np.expand_dims(new_image, axis=0)
prediction = model.predict(new_image)
if prediction[0][0] > 0.5:
    print('Prediction: Dog')
else:
    print('Prediction: Cat')
