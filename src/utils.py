"""
This file contains all the important utilities used frequently across the project
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize


def convert_data_to_csv(data,labels,filename):

    labels = labels.reshape((-1,1))
    c = np.concatenate((data[:, ::-1], labels), axis=1)
    print "Saving to %s.csv..."%filename
    np.savetxt(filename, c[:, ::-1], delimiter=" ")


def build_image(labels,shape):#pass orginal full rgb shape
    # print "recreating..."
    # print "input image shape is ",shape
    # print "input labels shape is ",labels.shape
    l,b,d = shape
    new_img = np.zeros(shape)


    # # labels = labels.reshape(shape)
    # unique_classes, which_classes = np.unique(labels,return_inverse=True)
    # print "unique_classes:",unique_classes
    # print "which_classes shape is",which_classes.shape
    # which_classes = which_classes.reshape(shape[0],shape[1])
    # # colors = create_colors(len(unique_classes))
    # colors = np.array([[255,0,0],[0,0,0],[0,255,0],[255,255,0]])
    # print "colors:",colors
    # for i in xrange(shape[0]):
    #     for j in xrange(shape[1]):
    #         # print "which_classes[%d][%d] ="%(i,j),which_classes[i][j]
    #         new_img[i][j] = colors[which_classes[i][j]]

    labels = labels.reshape((l,b))
    for i in xrange(l):
        for j in xrange(b):
            new_img[i][j] = {
                0:[255,0,0],
                1:[0,0,0],
                2:[0,255,0]
            }.get(labels[i][j],[255,255,0]) #return yellow as default if key not found


    return new_img

def create_colors(n_colors):
    values = range(256)
    colors = []
    for _ in xrange(n_colors):
        colors.append([np.random.choice(values) for __ in xrange(3)])
    return colors

def onehot(labels,custom_n_uniques=None):
    from keras.utils.np_utils import to_categorical
    uniques, ids = np.unique(labels,return_inverse=True)
    if custom_n_uniques is None:
        return to_categorical(ids,len(uniques))
    else:
        return to_categorical(ids,custom_n_uniques)

def inverse_onehot(matrix):
    return np.argmax(matrix,axis=1)

def show_images(*args,**kwargs):
    """

    :param args: list of images to be shown
    :param kwargs: 1. "titles": list of titles
    :return: None , Displays the images
    """

    for j,list_images in enumerate(args):
        for i,image in enumerate(list_images):
            if kwargs.get("titles") is None:
                plt.figure(i+j*100)
            else:
                plt.figure(kwargs.get("titles")[j][i])
            plt.clf()
            plt.imshow(image)
    plt.show()

def read_show_images(list_image_paths):#takes image path as input
    ims = []
    for path in list_image_paths:
        ims.append(plt.imread(path))
    show_images(ims)

def form_tuple_seeds(seed_length=2,total_numbers_domain=6):
    seeds = [(seed_length*i,seed_length*(i+1)-1) for i in xrange(total_numbers_domain/seed_length)]
    return seeds

def get_windows(nparr_image_data,nparr_label_image,inner_window_size=2,outer_window_size=4):
    print "getting windows..."
    inner_windows = []
    outer_windows = []
    diff = outer_window_size-inner_window_size
    seeds_rows = form_tuple_seeds(seed_length=inner_window_size, total_numbers_domain=nparr_image_data.shape[0])
    seeds_cols = form_tuple_seeds(seed_length=inner_window_size, total_numbers_domain=nparr_image_data.shape[1])
    nparr_padded = np.pad(nparr_image_data,((diff/2,diff/2),(diff/2,diff/2),(0,0)),'constant',constant_values=0)
    # print "padded array:\n",nparr_padded
    # print "orig has shape",nparr.shape
    # print "padded has shape :",nparr_padded.shape

    for m in seeds_rows:
        for n in seeds_cols:
            inner_ids = np.array(np.fromfunction(lambda i, j: (i >= m[0]) & (i <= m[1]) & (j >= n[0]) & (j <= n[1]), (nparr_image_data.shape[0],nparr_image_data.shape[1])))
            outer_ids = np.array(np.fromfunction(lambda i, j: (i >= m[0]) & (i <= m[1]+diff) & (j >= n[0]) & (j <= n[1]+diff), (nparr_padded.shape[0],nparr_padded.shape[1])))
            # print "shape:",nparr[inner_ids].shape[0]
            if nparr_image_data[inner_ids].shape[0]!=inner_window_size**2:
                print "actually it is :",nparr_image_data[inner_ids]

                continue
            inner_windows.append(nparr_label_image[inner_ids].reshape(inner_window_size,inner_window_size,-1))
            outer_windows.append(nparr_padded[outer_ids].reshape(outer_window_size,outer_window_size,-1))
    print "...done"
    return np.array(outer_windows),np.array(inner_windows)

def get_outer_windows(nparr_image_data,inner_window_size=16,outer_window_size=64):
    print "getting outer windows..."
    outer_windows = []
    diff = outer_window_size - inner_window_size
    seeds_rows = form_tuple_seeds(seed_length=inner_window_size, total_numbers_domain=nparr_image_data.shape[0])
    seeds_cols = form_tuple_seeds(seed_length=inner_window_size, total_numbers_domain=nparr_image_data.shape[1])
    nparr_padded = np.pad(nparr_image_data, ((diff / 2, diff / 2), (diff / 2, diff / 2), (0, 0)), 'constant',
                          constant_values=0)
    # print "padded array:\n",nparr_padded
    # print "orig has shape",nparr.shape
    # print "padded has shape :",nparr_padded.shape

    for m in seeds_rows:
        for n in seeds_cols:
            outer_ids = np.array(
                np.fromfunction(lambda i, j: (i >= m[0]) & (i <= m[1] + diff) & (j >= n[0]) & (j <= n[1] + diff),
                                (nparr_padded.shape[0], nparr_padded.shape[1])))
            # print "shape:",nparr[inner_ids].shape[0]
            outer_windows.append(nparr_padded[outer_ids].reshape(outer_window_size, outer_window_size, -1))
    print "...done"
    return np.array(outer_windows)

def onehot_images(list_images):
    onehot_imgs = []
    list_images = threshold_images(list_images)
    for image in list_images:
        l, b, d = image.shape
        x = np.zeros(( l, b, 4))
        x[(image == (255, 0, 0)).all(axis=2)] = [1, 0, 0,0]
        x[(image == (0, 0, 0)).all(axis=2)]= [0, 1, 0,0]
        x[(image == (0, 255, 0)).all(axis=2)]= [0, 0, 1,0]
        x[(image == (255, 255, 0)).all(axis=2)] = [0, 0, 0, 1]
        onehot_imgs.append(x)
    return onehot_imgs

def onehot_images_dash(list_images):
    onehot_imgs = []
    list_images = threshold_images(list_images)
    for image in list_images:
        l, b, d = image.shape
        x = np.zeros(( l, b, 2))
        x[(image == (255, 0, 0)).all(axis=2)] = [1, 0]
        x[(image == (0, 0, 0)).all(axis=2)]= [0, 1]
        x[(image == (0, 255, 0)).all(axis=2)]= [0, 1]
        onehot_imgs.append(x)
    return onehot_imgs


def threshold_images(list_images):
    for i in xrange(len(list_images)):
       list_images[i]= np.where(list_images[i]>170,255,0)
    return list_images

def inverse_onehot_images(list_label_images):
    inverted_label_images = []
    for label_image in list_label_images:
        print "label_image.shape",label_image.shape
        l,b,d = label_image.shape
        x = np.zeros((l,b,4))
        x[(label_image == [1,0,0,0]).all(axis=2)] = [255,0,0]
        x[(label_image == [0,1,0,0]).all(axis=2)] = [0,0,0]
        x[(label_image == [0,0,1,0]).all(axis=2)] = [0,255,0]
        x[(label_image == [0,0,0,1]).all(axis=2)] = [255,255,0]
        inverted_label_images.append(x)
    return inverted_label_images

def convert_cnn_preds_to_label_images(preds,shape=(16,16,4)):
    l,b,d = shape
    preds = np.array(preds)
    # preds = preds.reshape(-1,l,b,d)
    print "before, preds:",preds
    i_one_hotted_preds = np.argmax(preds,axis=3)
    print "ohp,",i_one_hotted_preds
    print "o.h.p shape",i_one_hotted_preds.shape
    label_imgs = []
    for pred in i_one_hotted_preds:
        label_img = build_image(pred,(l,b,3))
        label_imgs.append(label_img)
    return np.array(label_imgs)

# def convert_bce_to_imgs(preds,shape=(16,16,4)):
#     l,b,d = shape
#     preds = np.array(preds)
#     preds = preds.reshape(-1,l,b)
#     print "before, preds:",preds
#     i_one_hotted_preds = np.argmax(preds,axis=3)
#     print "ohp,",i_one_hotted_preds
#     print "o.h.p shape",i_one_hotted_preds.shape
#     label_imgs = []
#     for pred in i_one_hotted_preds:
#         label_img = build_image(pred,(l,b,3))
#         label_imgs.append(label_img)
#     return np.array(label_imgs)



def get_seed_window(x_shape,patch_size=16):
    m, n, d = x_shape
    R_col = np.arange(patch_size * d)
    R_row = np.arange(patch_size)
    seed_window = R_row[:, None] * n * d + R_col
    return seed_window

def get_rect_seed_window(x_shape,window_shape):
    m,n,d = x_shape
    l,b = window_shape
    R_col = np.arange(b*d)
    R_row = np.arange(l)
    seed_window = R_row[:,None]*n*d + R_col
    return seed_window

def get_magic(x_shape,patch_size):
    m, n, d = x_shape
    total_steps_cols = int(n/patch_size)
    total_steps_rows = int(m/patch_size)
    # print "total steps:",total_steps_rows,total_steps_cols
    col_inds = np.arange(0,(total_steps_cols-1)*patch_size+1,patch_size)
    row_inds = np.arange(0,(total_steps_rows-1)*patch_size+1,patch_size)
    magic = (row_inds[:,None]*n+col_inds).reshape(-1,1)[:,None]*3
    return magic

def get_patches(x,y,inner_patch_size=16,outer_patch_size=64):
    '''x stands for input numpy image array - for outer patches
        y stands for input numpy label image array - for inner label patches'''
    # print "getting patches..."
    diff = outer_patch_size - inner_patch_size
    inner_seed_window = get_seed_window(y.shape,inner_patch_size)
    inner_magic = get_magic(y.shape,inner_patch_size)
    inner_windows = np.take(y,inner_seed_window + inner_magic)
    l, b, h = inner_windows.shape

    inner_windows = inner_windows.reshape((l, b, -1, y.shape[2]))

    x_padded = np.pad(x, ((diff / 2, diff / 2), (diff / 2, diff / 2), (0, 0)), 'constant',
                              constant_values=0)
    outer_seed_window = get_seed_window(x_padded.shape,outer_patch_size)
    outer_windows = np.take(x_padded,outer_seed_window + inner_magic)
    l, b, h = outer_windows.shape

    outer_windows = outer_windows.reshape((l, b, -1, x.shape[2]))
    # print "...done"
    # print "inner_window.shape =", inner_windows.shape
    # print "outer_window.shape =", outer_windows.shape
    return inner_windows, outer_windows

def get_context_windows(x,window_size=5,random=False,random_sample_size=1000):
    from sklearn.utils import shuffle
    m,n,d = x.shape
    diff  = window_size - 1 # outer - inner size, see above method get_patches
    x_padded = np.pad(x, ((diff / 2, diff / 2), (diff / 2, diff / 2), (0, 0)), 'constant',
                      constant_values=0)
    # inner_magic = np.arange(m*n*d)[::3].reshape((-1,1,1))
    inner_window_magic = get_rect_seed_window(x_padded.shape,(m,n)).ravel()
    inner_magic = inner_window_magic[::3].reshape((-1,1,1))
    if random is True:
        inner_magic =shuffle(inner_magic)[:random_sample_size]
    outer_seed_window = get_seed_window(x_padded.shape,patch_size=window_size)
    # print outer_seed_window + inner_magic
    outer_windows = np.take(x_padded, outer_seed_window + inner_magic)
    l,b,h = outer_windows.shape
    outer_windows = outer_windows.reshape((l,b,-1,d))
    return outer_windows

def convert_ordering_th_to_tf(data):
    n, c,l,b = data.shape
    R_values = data[:, 0, :, :].reshape((-1,  l, b,1))
    G_values = data[:, 1, :, :].reshape((-1,  l, b,1))
    B_values = data[:, 2, :, :].reshape((-1,  l, b,1))
    print "shapes", R_values.shape, G_values.shape, B_values.shape
    data_dash = np.concatenate((R_values, G_values, B_values), axis=3)
    print data_dash.shape
    return data_dash

def convert_ordering_tf_to_th(data):
    n,l,b,c =  data.shape
    R_values = data[:, :, :, 0].reshape((-1, 1, l, b))
    G_values = data[:, :, :, 1].reshape((-1, 1, l, b))
    B_values = data[:, :, :, 2].reshape((-1, 1, l, b))
    print "shapes", R_values.shape, G_values.shape, B_values.shape
    data_dash = np.concatenate((R_values, G_values, B_values), axis=1)
    print data_dash.shape
    return data_dash

def separate_channels(x):
    x_dash = np.array(np.split(x,range(1,x.shape[-1]),axis=len(x.shape)-1))
    x_dash_dash = np.array(np.split(x_dash,range(1,x_dash.shape[1]),axis=1))
    y = x_dash_dash[:,:,0,:,:,0]
    return y

def combine_channels(x):
    #invert separation#
    x_ = np.stack(x,axis=1)
    y = np.stack(x_,axis=-1)
    print "combined shape:",y.shape
    return y


def save_model(model,model_name):
    import json
    ##save model and architecture##
    model.save("models/" + model_name + ".h5")
    model_json = model.to_json()
    with open("models/arch/" + model_name + ".json", "wb") as outfile:
        json.dump(model_json, outfile)
    model.save_weights("models/" + model_name + "_wts.h5")


def onehot_images_dash_dash(list_images):
    onehot_imgs = []
    list_images = threshold_images(list_images)
    for image in list_images:
        l, b, d = image.shape
        x = np.zeros(( l, b))
        x[(image == (255, 0, 0)).all(axis=2)] = 1
        x[(image == (0, 0, 0)).all(axis=2)]= 0
        x[(image == (0, 255, 0)).all(axis=2)]= 0
        onehot_imgs.append(x)
    return onehot_imgs



def prepare_result(orig_image,label_img,heat_map,**kwargs):
    from helper import Result
    # heat_map_metrics = kwargs.get("heat_map_metrics")
    # mask_metrics = kwargs.get("mask_metrics")
    img_t = np.where(heat_map>0.5,1,0)
    result = Result(orig_image,heat_map,img_t,label_img,**kwargs)
    return result



def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs, path

    try:
        makedirs(mypath)
    except OSError as exc:  # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            if exc.errno == EEXIST and path.isdir(mypath):
                ans = raw_input("Result data with this model name already exists, Overwrite data y/n: ")
                if ans == 'y':
                    return True
                else:
                    return False
        else:
            raise
    return True


def stitch(data,orig_image_cols,divisions):
    a = data
    print "a.shape",a.shape
    b = a.reshape(-1,divisions,*a[0].shape)
    print "b.shape",b.shape
    c = np.array([np.hstack(x) for x in b])
    print "c.shape",c.shape
    d = np.array([np.vstack(np.split(x,[orig_image_cols],axis=1)) for x in c])
    print "d.shape",d.shape
    return d

def divide_image(test_image):
    l,b = np.shape(test_image)[:2]
    t1, t2, t3, t4 = test_image[:l/2, :b/2], test_image[:l/2, b/2:b], test_image[l/2:l, :b/2], test_image[l/2:l, b/2:b]
    return np.array([t1,t2,t3,t4])

def resize_images(*args):
    resized = []
    for i in xrange(len(args)):
        resized.append(imresize(args[i],(1024,1024)))
    return tuple(resized)


def get_complete_result(input_image,input_gt,name):
    from helper import ResultBuilder, CompleteResult
    input_image , input_gt = resize_images(input_image,input_gt)
    builder = ResultBuilder(input_image,input_gt)
    complete_result = CompleteResult(name)
    complete_result.set_hfig(builder.build_hfig())
    complete_result.set_mfig(builder.build_mfig())
    complete_result.set_table(builder.build_table())
    complete_result.set_metrics(builder.build_metrics_dict())
    return complete_result



def save_fmaps():
    from helper import UFCN
    import glob
    from tqdm import tqdm

    u = UFCN()

    feature_maps = []
    for img in tqdm(glob.glob('data/uavdata/uav/*')):
        feature_map = u.take_out('activation_3', img, average=False)
        print "Shape is", np.shape(feature_map)
        feature_maps.append(feature_map)

    maps = []
    for fmap in feature_maps:
        fmap = combine_channels([fmap])[0]
        d = divide_image(fmap)
        fmaps = separate_channels(d)
        maps.extend(fmaps)

    np.save('data/fmaps.npy', np.array(maps))
    print 'final shape is', np.shape(maps)





