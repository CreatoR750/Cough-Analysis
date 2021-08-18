import os
import math
import argparse
import ffmpeg
import pywt
import csv
from sklearn import preprocessing
import scaleogram as scg
import numpy as np
import pandas as pd

from numpy import *
import scipy.io.wavfile as wavfile
from scipy import *
from pylab import *
from converter import convert
from test2 import wavelet, neuralNet,wavelet_solo, learn
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

formats_to_convert = ['.mp3']
dirpath= 'Sounds2/'
convertM4A=convert(formats_to_convert, dirpath)
#path='Sounds/Covid/1.wav'
#wave1= wavelet_solo(path)





#axes = scg.plot_wav('mexh', figsize=(14,5))
#show()



#wavelet1=wavelet(dir_name1,names1)
#wavelet2=wavelet(dir_name2,names2)
#wavelet3=wavelet(dir_name3,names3)
#wavelet4=wavelet(dir_name4,names4)
#wavelet5=wavelet(dir_name5,names5)
#wavelet6=wavelet(dir_name6,names6)
#wavelet7=wavelet(dir_name7,names7)
#wavelet8=wavelet(dir_name8,names8)
#learn1 = learn()

#NN = neuralNet()








