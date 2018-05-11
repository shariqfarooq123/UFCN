from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D,Deconvolution2D, Activation,Flatten
from keras.optimizers import Adam
from keras import backend as K
from utils import save_model,separate_channels,combine_channels,show_images
from keras.preprocessing.image import ImageDataGenerator

K.set_image_dim_ordering('th')

img_rows = 512
img_cols = 512
input_imp_depth = 3
n_classes = 2
smooth = 1.
init = 'glorot_normal'

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_model():
    inputs = Input((input_imp_depth, img_rows, img_cols))
    conv1 = Convolution2D(16, 3, 3,  border_mode='same',init=init)(inputs)
    conv1 = Convolution2D(16, 3, 3,  border_mode='same',init=init)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    act1 = Activation("relu")(pool1)

    conv2 = Convolution2D(32, 3, 3, border_mode='same',init=init)(act1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    act2 = Activation("relu")(pool2)

    conv3 = Convolution2D(64, 3, 3,  border_mode='same',init=init)(act2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    act3 = Activation("relu")(pool3)

    conv5 = Convolution2D(128, 3, 3,  border_mode='same',init=init)(act3)
    up7 = merge([Deconvolution2D(128,2,2,output_shape=(None,128,img_rows/4,img_cols/4),subsample=(2,2),init=init)(conv5), conv3], mode='concat', concat_axis=1)
    act7 = Activation("relu")(up7)
    conv7 = Convolution2D(64, 3, 3,  border_mode='same',init=init)(act7)

    up8 = merge([Deconvolution2D(64,2,2,output_shape=(None,64,img_rows/2,img_cols/2),subsample=(2,2),init='normal')(conv7), conv2], mode='concat', concat_axis=1)
    act8 = Activation("relu")(up8)
    conv8 = Convolution2D(32, 3, 3,  border_mode='same',init=init)(act8)
    conv8 = Convolution2D(32, 3, 3, border_mode='same',init=init)(conv8)

    up9 = merge([Deconvolution2D(32,2,2,output_shape=(None,32,img_rows,img_cols),subsample=(2,2),init='normal')(conv8), conv1], mode='concat', concat_axis=1)
    act9 = Activation("relu")(up9)
    conv9 = Convolution2D(16, 3, 3,  border_mode='same',init=init)(act9)
    conv9 = Convolution2D(16, 3, 3,  border_mode='same',init=init)(conv9)

    conv10 = Convolution2D(1, 1, 1,init=init,activation="sigmoid")(conv9)
    flatten1 = Flatten()(conv10)
    model = Model(input=inputs, output=flatten1)

    model.compile(optimizer=Adam(1e-5), loss="binary_crossentropy", metrics=[dice_coef,"accuracy"])

    return model

model = get_model()
print "summary:\n",model.summary()
from loader import load_bce_data
data , labels = load_bce_data()
shape = data[0].shape
print data.shape, labels.shape
data = separate_channels(data)
print data.shape , labels.shape
labels = labels.reshape((-1,1,img_rows,img_cols))


# we create two instances with the same arguments
data_gen_args = dict(
                     rotation_range=90.,
                     width_shift_range=0.1,
                     height_shift_range=0.1,
                     zoom_range=0.2)
image_datagen = ImageDataGenerator(**data_gen_args)
mask_datagen = ImageDataGenerator(**data_gen_args)

# Provide the same seed and keyword arguments to the fit and flow methods
seed = 1
image_datagen.fit(data, augment=True, seed=seed)
mask_datagen.fit(labels, augment=True, seed=seed)

image_generator = image_datagen.flow_from_directory(
    'data/uavdata/uav',
    class_mode='binary',
    seed=seed)

mask_generator = mask_datagen.flow_from_directory(
    'data/uavdata/gts',
    class_mode='binary',
    seed=seed)

# combine generators into one which yields image and masks
train_generator = zip(image_generator, mask_generator)

model.fit_generator(
    train_generator,
    samples_per_epoch=2000,
    nb_epoch=50)
#
# from itertools import izip
# nb_epoch = 20
# for e in range(nb_epoch):
#     print 'Epoch', e
#     batches = 0
#     for X_batch, Y_batch in izip(image_datagen.flow(data,shuffle=False,batch_size=1,seed=seed),mask_datagen.flow(labels,shuffle=False,batch_size=1,seed=seed)):
#         print X_batch.shape, Y_batch.shape
#         x = combine_channels([X_batch[0]])[0]
#         y = Y_batch[0].reshape((img_rows,img_cols))
#         print x.shape, y.shape
#         # show_images([x,y])
#         #
#         # raise
#         loss = model.train(X_batch, Y_batch)
#         batches += 1
#         if batches >= len(X_train) / 32:
#             # we need to break the loop by hand because
#             # the generator loops indefinitely
#             break




save_model(model,"aug_dunet")

from sklearn.utils import shuffle
inds = shuffle(range(len(data)))[:5]
preds = model.predict(data[inds],verbose=1)
print "preds orig:\n",preds
print "orig shape",preds.shape
data_dash = combine_channels(data[inds])
preds = preds.reshape((-1,img_rows,img_cols))
show_images(data_dash,preds/255)

