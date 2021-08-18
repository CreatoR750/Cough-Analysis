import os
import pyaudio
import wave
import scaleogram as scg
import numpy as np
import pandas as pd
from numpy import *
import scipy.io.wavfile as wavfile
from scipy import *
from pylab import *
from converter import convert
from wavelet import wavelet
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K


def wavelet(dir_name,names):
    for name in names:
        fullname = os.path.join(dir_name, name)
        (rate, X) = wavfile.read(fullname)
        name_split = fullname.split('/')
        length = X.shape[0] / rate
        time = np.linspace(0., length, X.shape[0])
    # plt.plot(time, X[:, 0], label="Left channel")
    # plt.plot(time, X[:, 1], label="Right channel")
    # plt.legend()
    # plt.xlabel("Time [s]")
    # plt.ylabel("Amplitude")
    # plt.show()

        X = X[:, :1]
    # plot(X)
        data = pd.DataFrame(data=X, columns=['JJJ'], dtype=float)
        data1 = data.values.squeeze()
    # print(min(data1))
    # print(max(data1))
        N = data1.size
        t0 = 0;
        dt = 0.00002
        time = t0 + arange(len(data1)) * dt
        wavelet = 'cmor1-0.5'
        scales = scg.periods2scales(np.arange(1, 32))
        ax = scg.cws(time, data1, scales=scales, wavelet=wavelet,
                     cmap="jet", cbar=None, xlabel="", yscale="linear",
                     title='')
        #savefig('Plots/Covid/sample1.png')
        pad_inches = 0
        axis('off')
        #os.mkdir('Plots/'+name_split[1])

        #savefig('Plots/' + str(name_split[1])+'/' + name + '.png', bbox_inches='tight',  pad_inches = 0)
        savefig('data/test' + name + '.png', bbox_inches='tight', pad_inches=0)
        #show()

#(rate, X) = wavfile.read('sounds/sample0.wav')
#print(X)
#X=X[X!=0]
#print(X)

#ticks = ax.set_yticklabels([2,4,8, 16,32])


def wavelet_solo(fullname):

    (rate, X) = wavfile.read(fullname)
    name_split = fullname.split('/')
    length = X.shape[0] / rate
    time = np.linspace(0., length, X.shape[0])
    plt.figure(figsize=(6, 2.5))
    plt.plot(time, X)
    plt.xlabel("Time [s]")
    #plt.ylabel("Amplitude")
    plt.title('Звук')


    plt.savefig('data/test/sound.png', dpi=90)
    if X.ndim == 2:
        X = X[:, :1]
    else:
        print(X.ndim)

    data = pd.DataFrame(data=X, columns=['JJJ'], dtype=float)
    data1 = data.values.squeeze()

    N = data1.size
    t0 = 0;
    dt = 0.00002
    time = t0 + arange(len(data1)) * dt
    wavelet = 'cmor1-0.5'
    scales = scg.periods2scales(np.arange(1, 32))
    ax = scg.cws(time, data1, scales=scales, wavelet=wavelet,
                    cmap="jet", cbar=None, xlabel="Время в сек", yscale="linear",
                    title='Вейвлет преобразование')
    savefig('data/test/show_test.png')
    title('')
    axis('off')
    savefig('data/test/test.png', bbox_inches='tight', pad_inches=0)


#(rate, X) = wavfile.read('sounds/sample0.wav')
#print(X)
#X=X[X!=0]
#print(X)

#ticks = ax.set_yticklabels([2,4,8, 16,32])



def learn():
    width, height = 496, 369
    train_dir = 'data/train1/'
    validation_dir = 'data/validation1/'
    train_samples = 269
    validation_samples = 158
    epochs = 25
    batch_size = 32

    if K.image_data_format() == 'channels_first':
        input_shape = (3, width, height)
    else:
        input_shape = (width, height, 3)

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation('relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_data = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(width, height),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_data.flow_from_directory(
        validation_dir,
        target_size=(width, height),
        batch_size=batch_size,
        class_mode='categorical')

    #history = model.fit(
        #train_generator,
        #steps_per_epoch=train_samples // batch_size,
        #epochs=epochs,
        #validation_data=validation_generator)
        #validation_steps=validation_samples // batch_size)
    #model.save_weights('data/new_weights_1net.h5')
    #acc = history.history['accuracy']
    #val_acc = history.history['val_accuracy']

    # = history.history['loss']
    #val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    #plt.figure()
    #plt.subplot(1, 2, 1)
    #plt.plot(epochs_range, acc, label='Точность на обучении')
    #plt.plot(epochs_range, val_acc, label='Точность на валидации')
    #plt.legend(loc='lower right')
    #plt.title('Точность на обучающих и валидационных данных')
    #plt.show()
    #plt.savefig('acc.png')
    #plt.subplot(1, 2, 2)
    #plt.subplot(1, 2, 2)
    #plt.plot(epochs_range, loss, label='Потери на обучении')
    #plt.plot(epochs_range, val_loss, label='Потери на валидации')
    #plt.legend(loc='upper right')
    #plt.title('Потери на обучающих и валидационных данных')
    #plt.savefig('loss.png')
    #plt.show()

    model.load_weights('data/new_weights_1net.h5')
    scores = model.evaluate(train_generator, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

def neuralNet(path):
    # 1 neural net
    # dimensions of our images.
    img_width, img_height = 496, 369

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    model_1 = Sequential()
    model_1.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model_1.add(Activation('relu'))
    model_1.add(MaxPooling2D(pool_size=(2, 2)))

    model_1.add(Conv2D(64, (3, 3)))
    model_1.add(Activation('relu'))
    model_1.add(MaxPooling2D(pool_size=(2, 2)))

    model_1.add(Conv2D(64, (3, 3)))
    model_1.add(Activation('relu'))
    model_1.add(MaxPooling2D(pool_size=(2, 2)))

    model_1.add(Flatten())
    model_1.add(Dense(256))
    model_1.add(Activation('relu'))
    #model_1.add(Dropout(0.5))
    model_1.add(Dense(2))
    model_1.add(Activation('softmax'))

    model_1.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

    # 2 neural net
    model_2 = Sequential()
    model_2.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model_2.add(Activation('relu'))
    model_2.add(MaxPooling2D(pool_size=(2, 2)))

    model_2.add(Conv2D(64, (3, 3)))
    model_2.add(Activation('relu'))
    model_2.add(MaxPooling2D(pool_size=(2, 2)))

    model_2.add(Conv2D(64, (3, 3)))
    model_2.add(Activation('relu'))
    model_2.add(MaxPooling2D(pool_size=(2, 2)))

    model_2.add(Flatten())
    model_2.add(Dense(256))
    model_2.add(Activation('relu'))
    #model_2.add(Dropout(0.5))
    model_2.add(Dense(4))
    model_2.add(Activation('softmax'))

    model_2.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
    # Prediction
    class_list_1 = os.listdir('data/validation1')
    class_list_1 = sorted(class_list_1)
    class_list_2 = os.listdir('data/validation3')
    class_list_2 = sorted(class_list_2)

    model_1.load_weights('data/new_weights_1net.h5')
    model_2.load_weights('data/new_weights_2net.h5')

    img = image.load_img(path, target_size=(496, 369))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x * 1. / 255  # rescale as training

    prediction_1 = model_1.predict(x)
    print(prediction_1)
    print(class_list_1[np.argmax(prediction_1)])

    if class_list_1[np.argmax(prediction_1)] == 'Cough':

        print("That's a cough")
        result1= "That's a cough."
        prediction_2 = model_2.predict(x)
        print(prediction_2)
        print(class_list_2[np.argmax(prediction_2)])
        result2 = class_list_2[np.argmax(prediction_2)]
    else:
        print("You selected not a cough")
        result1="You selected not a cough."
        result2=''

    result = result1+' '+'Prediction: ' + result2
    print(result)

    return result

def record():

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 2
    WAVE_OUTPUT_FILENAME = "RecordedSounds/output.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("* done recording")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()




