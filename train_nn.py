import numpy as np
import pickle
import cv2
from sklearn.model_selection import train_test_split
import pickle
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import layers

data = pickle.loads(open('output/embeddings.pickle', "rb").read())
x = d = np.array(data["embeddings"])
y = pickle.loads(open('./output/le.pickle', "rb").read()).fit_transform(data["names"])


x_train, x_test, y_train, y_test = train_test_split(x, y)

train_images = x_train / 255.0
test_images = x_test / 255.0

print(train_images.shape)
print(test_images.shape)


#
model = tf.keras.Sequential([
    # tf.keras.layers.Flatten(input_shape=(11, 11)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(14)
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_images, y_train, epochs=120)

test_loss, test_acc = model.evaluate(test_images,  y_test, verbose=2)

print('\nTest accuracy:', test_acc)

model.save('output/nn_model')

# f = open('output/', "wb")
# f.write(pickle.dumps(model))
# f.close()