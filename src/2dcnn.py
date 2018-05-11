from keras.models import load_model, Model, Sequential
from keras.layers import Input, Dense, Flatten, Reshape
from keras import backend as K
from scipy.misc import imresize
from loader import load_bce_data
from utils import separate_channels, combine_channels
import numpy as np

K.set_image_dim_ordering('th')  # set Theano dimension ordering in this code

img_rows = 512  # Input image rows
img_cols = 512  # Input image columns
input_imp_depth = 3  # Input image feature maps, here 3 for R, G & B
n_classes = 2  # Total number of predicted classes // Here 2, Road & Background
smooth = 1.  # Smoothness parameter in the below dice coefficient function
init = 'glorot_normal'  # Initialize weights with random values taken from a normal distribution, glorot way


def get_model():

    model = Sequential([
        Flatten(input_shape=(64,64,64)),
        Dense(4096,activation='sigmoid'),
        Reshape((64,64))
    ])
    model.compile('adam','binary_crossentropy',['accuracy'])
    return model


def load_data():
    print 'loading training data...'
    imgs, gts = load_bce_data()
    gts = [imresize(gt,(64,64)) for gt in gts]
    print '...done'
    imgs = separate_channels(imgs)
    return imgs,gts

def load_train():
    imgs, gts = load_data()
    fmaps = np.load('data/fmaps.npy')
    return fmaps, gts


def predict(model=None):
    imgs, gts = load_train()
    print 'loading model...'
    if not model:
        model = get_model()
    print 'predicting...'
    preds = model.predict(imgs, verbose=1, batch_size=4)
    return preds

def train():
    imgs, gts = load_train()
    print 'loading model...'
    model = get_model()
    print 'fitting...'
    model.fit(np.array(imgs),np.array(gts),batch_size=4)
    print '...done'
    model.save('models/2dcnn.h5')
    return model


if __name__ == '__main__':
    model = train()
    preds = predict(model)

