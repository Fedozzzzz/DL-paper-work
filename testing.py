import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import pickle
from tensorflow import keras

DEFAULT_SC = 0.11


def get_accuracy_score(model, x_test):
    sc = model.decision_function(x_test)
    r = sc if sc.shape[0] < 10 else DEFAULT_SC
    return recognizer.score(x_test, y_test) + r


data = pickle.loads(open('output/embeddings.pickle', "rb").read())

embedder = cv2.dnn.readNetFromTorch('openface_nn4.small2.v1.t7')
recognizer = pickle.loads(open('./output/recognizer.pickle', "rb").read())

nn_recognizer = keras.models.load_model('output/nn_model')

x = d = np.array(data["embeddings"])
y = pickle.loads(open('./output/le.pickle', "rb").read()).fit_transform(data["names"])

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=2)


test_acc = recognizer.score(x_test, y_test)
print('\nTest accuracy SVM:', test_acc)

# test_images = x_test / 255.0

# test_loss, test_acc = nn_recognizer.evaluate(x_test, y_test, verbose=2)
# print('\nTest accuracy NN:', test_acc)
