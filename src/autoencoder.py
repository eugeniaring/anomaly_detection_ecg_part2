import keras
from keras import layers
from keras.models import Sequential


def define_ae(input_shape=(1,)):
    Autoencoder = Sequential()
    Autoencoder.add(keras.Input(shape=input_shape))
    Autoencoder.add(layers.Dense(16, activation="relu"))
    Autoencoder.add(layers.Dense(32, activation="relu"))
    Autoencoder.add(layers.Dense(input_shape[0], activation="sigmoid"))
    print('Autoencoder architecture: \n')
    print(Autoencoder.summary())  
    return Autoencoder