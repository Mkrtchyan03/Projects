import tensorflow as tf
import keras
from keras.layers import Dense, Conv2D, MaxPool2D, BatchNormalization, Flatten
import keras.regularizers as regularizers
from keras.optimizers import SGD, Adam
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


from numpy.core.fromnumeric import shape
class vggcifar100:
  def __init__(self, train=True):
    self.num_classes = 100
    self.weight_decay= 0.005
    self.x_shape = [32, 32, 3]

    self.model = self.build_model()
    if train:
      self.model = self.train(self.model)
    else:
      self.model.load_weights('/home/gagik/Documents/vgg/weights/cifar100vgg.h5')

  def build_model(self):
    model = tf.keras.Sequential()

    model.add(Conv2D(64, (3,3), padding='same',
                     input_shape=self.x_shape,
                     kernel_regularizer=regularizers.l2(self.weight_decay),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), padding='same',
                     kernel_regularizer=regularizers.l2(self.weight_decay),
                     activation='relu'))

    model.add(MaxPool2D(pool_size=(2,2), strides=2))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3), padding='same',
                     kernel_regularizer=regularizers.l2(self.weight_decay),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3), padding='same',
                     kernel_regularizer=regularizers.l2(self.weight_decay),
                     activation='relu'))

    model.add(MaxPool2D(pool_size=(2,2), strides=2))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3,3), padding='same',
                     kernel_regularizer=regularizers.l2(self.weight_decay),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3,3), padding='same',
                     kernel_regularizer=regularizers.l2(self.weight_decay),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3,3), padding='same',
                     kernel_regularizer=regularizers.l2(self.weight_decay),
                     activation='relu'))

    model.add(MaxPool2D(pool_size=(2,2), strides=2))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3,3), padding='same',
                     kernel_regularizer=regularizers.l2(self.weight_decay),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3,3), padding='same',
                     kernel_regularizer=regularizers.l2(self.weight_decay),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3,3), padding='same',
                     kernel_regularizer=regularizers.l2(self.weight_decay),
                     activation='relu'))

    model.add(MaxPool2D(pool_size=(2,2), strides=2))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3,3), padding='same',
                     kernel_regularizer=regularizers.l2(self.weight_decay),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3,3), padding='same',
                     kernel_regularizer=regularizers.l2(self.weight_decay),
                     activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3,3), padding='same',
                     kernel_regularizer=regularizers.l2(self.weight_decay),
                     activation='relu'))

    model.add(MaxPool2D(pool_size=(2,2), strides=2))
    model.add(Flatten())
    
    model.add(BatchNormalization())
    model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(self.weight_decay)))
    model.add(Dense(self.num_classes, activation='softmax'))

    return model

  def normalize(self, x_train, x_test):
    #this function normalize inputs for zero mean and unit variance

    mean = np.mean(x_train, axis=(0,1,2,3))
    std = np.std(x_train, axis=(0,1,2,3))
    x_train = (x_train-mean)/(std+1e-7)
    x_test = (x_test-mean)/(std+1e-7)

    return x_train, x_test

  def norm_prod(self, x):
    #this function is used to normalize instances in production according to saved training set statistics
    # Input: X - a training set
    mean = 121.936
    std = 68.389
    return (x-mean)/(std+1e-7)

  def predict(self, x, normalize=True, batch_size=50):
    if normalize:
      x = self.norm_prod(x)
    return self.model.predict(x, batch_size)

  def train(self, model):

    #train parametrs
    batch_size = 128
    learning_rate = 0.01
    lr_drop = 20
    lr_decay = 1e-6


    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train, x_test = self.normalize(x_train, x_test)

    y_train = tf.keras.utils.to_categorical(y_train, self.num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, self.num_classes)

    def lr_schedule(epoch):
      return learning_rate * (0.5 ** (epoch // lr_drop))
    reduce_lr = keras.callbacks.LearningRateScheduler(lr_schedule)

    #data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='/home/gagik/Documents/vgg/logs/batch_norm_sgd')

    sgd = SGD(learning_rate=learning_rate, nesterov=True, momentum=0.9)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    history = model.fit(x_train, y_train, batch_size=batch_size, validation_data=(x_test, y_test), callbacks=[tensorboard_callback], epochs=20)
    model.save_weights("/home/gagik/Documents/vgg/weights/cifar100vgg.h5")
    return model
