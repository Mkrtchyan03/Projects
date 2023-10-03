import keras
import numpy as np
import tensorflow as tf
from vgg import vggcifar100

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

y_train = keras.utils.to_categorical(y_train, 100)
y_test = keras.utils.to_categorical(y_test, 100)

model = vggcifar100()

predicted_x = model.predict(x_test)
residuals = (np.argmax(predicted_x,1)==np.argmax(y_test,1))
acc = sum(residuals)/len(residuals)

acc5 = (np.argmax(y_test,1).reshape((y_test.shape[0], 1)) == np.argsort(predicted_x, axis = 1)[:,-5:]).sum(axis=1).sum()/len(residuals)
print("the validation 0/1 accuracy top1 is: ",acc)
print("the validation 0/1 accuracy top5 is: ",acc5)

