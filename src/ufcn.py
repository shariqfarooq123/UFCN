from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, Deconvolution2D, Activation, Flatten
from keras.optimizers import Adam
from keras import backend as K
from utils import save_model, separate_channels, combine_channels, show_images
from loader import load_bce_data
from sklearn.model_selection import train_test_split

K.set_image_dim_ordering('th')  # set Theano dimension ordering in this code

img_rows = 512  # Input image rows
img_cols = 512  # Input image columns
input_imp_depth = 3  # Input image feature maps, here 3 for R, G & B
n_classes = 2  # Total number of predicted classes // Here 2, Road & Background
smooth = 1.  # Smoothness parameter in the below dice coefficient function
init = 'glorot_normal'  # Initialize weights with random values taken from a normal distribution, glorot way


# Dice coef is the measure of similarity between True segmentation and predicted one ,
# Used here just to get the idea about model performance
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_model():
    """
    This function defines the architecture of model
    :return: model
    """

    inputs = Input((input_imp_depth, img_rows, img_cols))
    conv1 = Convolution2D(16, 3, 3, border_mode='same', init=init)(
        inputs)  # Convolution layer with 16 filters , kernel size - 3x3, border_mode='same' implies input is padded with zeros so as to create the output of same spatial shape as input
    conv1 = Convolution2D(16, 3, 3, border_mode='same', init=init)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    act1 = Activation("relu")(
        pool1)  # Activation and pooling operations commute as long as activation function is an increasing funtion, so order of pooling and activation does not matter

    conv2 = Convolution2D(32, 3, 3, border_mode='same', init=init)(act1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    act2 = Activation("relu")(pool2)

    conv3 = Convolution2D(64, 3, 3, border_mode='same', init=init)(act2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    act3 = Activation("relu")(pool3)

    conv5 = Convolution2D(128, 3, 3, border_mode='same', init=init)(act3)

    deconv7 = merge([Deconvolution2D(128, 2, 2, output_shape=(None, 128, img_rows / 4, img_cols / 4), subsample=(2, 2),
                                 init=init)(conv5), conv3], mode='concat', concat_axis=1)  # Skip connection
    act7 = Activation("relu")(deconv7)
    conv7 = Convolution2D(64, 3, 3, border_mode='same', init=init)(act7)

    deconv8 = merge([Deconvolution2D(64, 2, 2, output_shape=(None, 64, img_rows / 2, img_cols / 2), subsample=(2, 2),
                                 init='normal')(conv7), conv2], mode='concat', concat_axis=1)
    act8 = Activation("relu")(deconv8)
    conv8 = Convolution2D(32, 3, 3, border_mode='same', init=init)(act8)
    conv8 = Convolution2D(32, 3, 3, border_mode='same', init=init)(conv8)

    deconv9 = merge(
        [Deconvolution2D(32, 2, 2, output_shape=(None, 32, img_rows, img_cols), subsample=(2, 2), init='normal')(conv8),
         conv1], mode='concat', concat_axis=1)
    act9 = Activation("relu")(deconv9)
    conv9 = Convolution2D(16, 3, 3, border_mode='same', init=init)(act9)
    conv9 = Convolution2D(16, 3, 3, border_mode='same', init=init)(conv9)

    conv10 = Convolution2D(1, 1, 1, init=init, activation="sigmoid")(conv9)
    flatten1 = Flatten()(conv10)  # Can use global average pooling instead of flatten to further avoid overfitting
    model = Model(input=inputs, output=flatten1)

    model.compile(optimizer=Adam(1e-5), loss="binary_crossentropy", metrics=[dice_coef, "accuracy"])

    return model


if __name__ == '__main__':
    model = get_model()
    print "summary:\n", model.summary()

    data, labels = load_bce_data()

    shape = data[0].shape
    print data.shape, labels.shape
    data = separate_channels(data)  # make data compatible with theano dimension ordering
    print data.shape, labels.shape
    labels = labels.reshape((-1,
                             img_rows * img_cols))  # Flatten the ground truths from 2D to 1D for evaluation of binary cross entropy at the other end of model
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels,
                                                                        test_size=0.3)  # split the data into 70-30 train-test
    model.fit(data_train, labels_train, batch_size=32, nb_epoch=3000)  # Train the model on the train-data portion
    save_model(model, "dunet7")  # Save the model

    from sklearn.utils import shuffle

    inds = shuffle(range(len(data_test)))[:5]
    preds = model.predict(data_test[inds],
                          verbose=1)  # Test the model on randomly picked 5 images from the test-portion
    print "preds orig:\n", preds
    print "orig shape", preds.shape
    data_dash = combine_channels(data_test[inds])  # Make data compatible for visualizing
    preds = preds.reshape((-1, img_rows, img_cols))  # Reshape back the predictions from flattened array to 2D
    show_images(data_dash, preds / 255)
