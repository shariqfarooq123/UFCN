import os, struct
from array import array as pyarray
from numpy import  array, int8, uint8, zeros
import numpy as np
import random
import matplotlib.pyplot as plt
import glob

def load_mnist(dataset="training", digits=np.arange(10), path="."):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'train-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 'train-labels.idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 't10k-images.idx3-ubyte')
        fname_lbl = os.path.join(path, 't10k-labels.idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()

    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels

def load_cfir(size=1000):
    from cPickle import load as cpLoad
    fo = open('data_batch_1','rb')
    data_dict = cpLoad(fo)
    fo.close()
    features = np.array(data_dict["data"])
    labels = np.array(data_dict["labels"])
    rand_ind = random.sample(range(len(features)),size)
    features,labels =  features[rand_ind], labels[rand_ind]
    # print "len(features):",len(features)
    # print "len(featues[0]):",len(features[0])
    from preprocess import rgb_major_transform_dataset
    features = rgb_major_transform_dataset(features)
    # features = rgb_to_gray_dataset(features)
    return features, labels

def load_road():
    from sklearn.utils import shuffle
    road = shuffle(plt.imread("data/roadr.jpg").reshape(-1, 3))
    road1 = shuffle(plt.imread("data/road1.jpg").reshape(-1, 3))[:10000]

    # road2 = shuffle(plt.imread("data/road2.jpg").reshape(-1, 3))[:10000]
    # not_road = shuffle(plt.imread("data/not_road.jpg").reshape(-1, 3))[:500]
    not_road1 = shuffle(plt.imread("data/not_roads.jpg").reshape(-1, 3))[:30000]
    not_road2 = shuffle(plt.imread("data/not_road2.jpg").reshape(-1, 3))[:20000]
    tree = shuffle(plt.imread("data/tree.jpg")).reshape(-1,3)
    tree2 = shuffle(plt.imread("data/tree_pallete1.jpg")).reshape(-1,3)[:50000]
    muddyroad = shuffle(plt.imread("data/muddyroad.jpg")).reshape(-1,3)
    # print tree.shape
    # tree = shuffle(plt.imread("data/tree.jpg")).reshape(-1,3)
    data_road = np.concatenate((road,
                                road1,
                                # road2,
                                not_road1,
                                not_road2,
                                tree,
                                tree2,
                                muddyroad))


    labels_road = np.concatenate((np.array([1 for i in xrange(len(road)+10000)]),
                                  np.array([2 for j in xrange(50000)]),
                                  np.array([3 for k in xrange(len(tree)+len(tree2))]),
                                  np.array([4 for _ in xrange(len(muddyroad))])
                                  ))
    return data_road, labels_road

def load_data_images(inner_window_size=16):
    from utils import onehot_images,inverse_onehot_images
    data_image_list = []
    label_image_list = []
    print "loading images..."
    for img_filename in glob.glob("data/manual_data/*.jpg"):
        img = plt.imread(img_filename)
        # print "shape is ",img.shape
        l,b,d = img.shape
        img = img[:l-l%inner_window_size,:b-b%inner_window_size] # make img dims divisible by 16
        data_image_list.append(img)
    print "loading label images..."
    for label_filename in glob.glob("data/manual_data_gt/*.jpg"):
        label_img = plt.imread(label_filename)
        l,b,d = label_img.shape
        label_img = label_img[:l-l%inner_window_size,:b-b%inner_window_size]
        label_image_list.append(label_img)
    print "done"
    # print "shape is ",np.array(label_image_list)
    label_image_list_onehot = onehot_images(label_image_list)
    return np.array(data_image_list),np.array(label_image_list_onehot)

def load_patched_data(inner_window_size=16,outer_window_size=64):
    print "patching..."
    from utils import get_patches
    data_windows_outer = []
    label_windows_inner = []
    images, labels = load_data_images(inner_window_size=inner_window_size)
    print "orig[0].shape:",images[0].shape, labels[0].shape
    for image,label in zip(images,labels):
        # outer, inner = get_windows(image,label,inner_window_size=inner_window_size,outer_window_size=outer_window_size)
        inner, outer = get_patches(image,label,inner_window_size,outer_window_size)
        # for outer_window,inner_window in zip(outer,inner):
        data_windows_outer.extend(outer)
        label_windows_inner.extend(inner)
    print "patching completed"
    return np.array(data_windows_outer),np.array(label_windows_inner)

def load_patched_data_from_pickle(filename="models/srdata.pkl"):
    '''

    :return:64 by 64 by 3 window data images
            and one hotted label images
    '''
    print "loading pickle..."
    import cPickle
    with open(filename, "rb") as file:
        data_dict = cPickle.load(file)
    print "...done"
    return data_dict["data"],data_dict["labels"]


# For pixel level approach
def load_pixel_context_data(window_size=5):
    from utils import get_context_windows

    #load images
    road_image1 = plt.imread("data/road.jpg")
    road_image3 = plt.imread("data/roadr.jpg")
    road_image2 = plt.imread("data/road1.jpg")
    road_image4 = plt.imread("data/road_palette2.jpg")
    road_image5 = plt.imread("data/road5.jpg")

    background1 = plt.imread("data/not_roads.jpg")
    background21 = plt.imread("data/back2_1.jpg")
    background22 = plt.imread("data/back2_2.jpg")
    background23 = plt.imread("data/back2_3.jpg")
    background24 = plt.imread("data/back2_4.jpg")
    background25 = plt.imread("data/back2_5.jpg")
    background26 = plt.imread("data/back2_6.jpg")
    # background2 = plt.imread("data/not_road2.jpg")
    background11 = plt.imread("data/re_back1.jpg")
    background111 = plt.imread("data/back1_1.jpg")
    background12 = plt.imread("data/back6.jpg")
    background13 = plt.imread("data/back1_3.jpg")

    veg1 = plt.imread("data/tree_test.jpg")
    veg2 = plt.imread("data/tree.jpg")
    veg3 = plt.imread("data/tree3.jpg")

    muddy_road1 = plt.imread("data/muddyroad.jpg")
    muddy_road2 = plt.imread("data/muddy_road2.jpg")

    #load windows
    road1_windows = get_context_windows(road_image1, window_size=window_size, random=True, random_sample_size=30000)
    road3_windows = get_context_windows(road_image3, window_size=window_size)
    road5_windows = get_context_windows(road_image5, window_size=window_size)
    # road4_windows = get_context_windows(road_image4, window_size=window_size)
    road2_windows = get_context_windows(road_image2,window_size=window_size,random=True,random_sample_size=30000)
    # road3_windows = get_context_windows(muddy_road2,window_size=window_size,random=True,random_sample_size=100)

    background1_windows = get_context_windows(background1,window_size=window_size,random=True,random_sample_size=5000)
    background21_windows = get_context_windows(background21,window_size=window_size,random=True,random_sample_size=5000)
    background111_windows = get_context_windows(background111,window_size=window_size)
    background22_windows = get_context_windows(background22,window_size=window_size,random=True,random_sample_size=5000)
    background23_windows = get_context_windows(background23,window_size=window_size,random=True,random_sample_size=5000)
    background24_windows = get_context_windows(background24,window_size=window_size,random=True,random_sample_size=5000)
    background25_windows = get_context_windows(background25,window_size=window_size,random=True,random_sample_size=5000)
    background26_windows = get_context_windows(background26,window_size=window_size,random=True,random_sample_size=5000)
    background12_windows = get_context_windows(background12,window_size=window_size,random=True, random_sample_size=5000)
    background13_windows = get_context_windows(background12,window_size=window_size)
    background11_windows = get_context_windows(background11,window_size=window_size,random=True,random_sample_size=30000)

    veg1_windows = get_context_windows(veg1,window_size=window_size,random=True,random_sample_size=20000)
    veg2_windows = get_context_windows(veg2,window_size=window_size)
    veg3_windows = get_context_windows(veg3,window_size=window_size)
    # muddy_road2_windows = get_context_windows(muddy_road2,window_size=window_size,random=True,random_sample_size=10000)
    # muddy_road1_windows = get_context_windows(muddy_road1,window_size=window_size)


    data = np.concatenate((road1_windows,
                           road2_windows,
                           road3_windows,
                           road5_windows,
                           # road4_windows,
                           # road3_windows,

                           background1_windows,
                           background21_windows,
                           background111_windows,
                           background22_windows,
                           background23_windows,
                           background24_windows,
                           background25_windows,
                           background26_windows,
                           background11_windows,
                           background12_windows,
                           background13_windows,

                           veg1_windows,
                           veg2_windows,
                           veg3_windows,

                           # muddy_road2_windows,
                           # muddy_road1_windows
                           ))
    total_road =    len(road1_windows)+\
                    len(road3_windows)+\
                    len(road5_windows)+ \
                    len(road2_windows)
                    # len(road4_windows)+\
    total_background = 60000+\
                        len(background12_windows)+\
                        len(background111_windows)+\
                        len(background13_windows)+\
                        len(background26_windows)

    total_veg = 20000+\
                 len(veg2_windows)+\
                 len(veg3_windows)

    labels = np.concatenate((
                            np.full((total_road ,),0),

                            np.full((total_background ,),1),

                            np.full((total_veg,),2),

                            # np.full((10000
                            #          +
                            #          len(muddy_road1_windows)
                            #          ,),3),

                            ))

    print "data.shape, labels.shape",data.shape,labels.shape
    print "\n0:road\n1:background\n2:vegetation\n3:muddy"
    print "total_road:",total_road
    print "total_background:",total_background
    print "total_veg:",total_veg
    return data,labels

def load_ccpgt_data(window_size=5):
    from utils import get_context_windows, onehot_images
    image = plt.imread("data/new10.jpg")
    data = get_context_windows(image,window_size=window_size)

    label_image = plt.imread("data/newgt10.jpg")
    label_image_onehotted = onehot_images([label_image])
    label_image_onehotted = label_image_onehotted[0].reshape((-1,4))

    return data,label_image_onehotted


def load_fcn_data(window_size=16):
    from utils import get_patches,onehot_images
    img = plt.imread("data/image_test10.jpg")
    label_img = plt.imread("data/10gt.jpg")
    label_img = onehot_images([label_img])
    data,labels = get_patches(img,label_img[0],inner_patch_size=window_size,outer_patch_size=window_size)
    return data,labels

def load_retrain_data():
    from utils import get_context_windows
    img = plt.imread("data/re_back1.jpg")
    data = get_context_windows(img)
    labels = np.full((len(data),3),[0,1,0])
    return data, labels



def load_new_data(window_size=5):
    from utils import get_context_windows
    road_data = []
    for img_file in glob.glob("data/rdata/road/*.jpg"):
        img = plt.imread(img_file)
        windows = get_context_windows(img,window_size=window_size)
        road_data.extend(windows)
    back_data = []
    for img_file in glob.glob("data/rdata/back/*.jpg"):
        img = plt.imread(img_file)
        windows = get_context_windows(img,window_size=window_size)
        back_data.extend(windows)
    veg_data = []
    for img_file in glob.glob("data/rdata/veg/*.jpg"):
        img = plt.imread(img_file)
        windows = get_context_windows(img,window_size=window_size)
        veg_data.extend(windows)

    road_data, back_data, veg_data = np.array(road_data), np.array(back_data), np.array(veg_data)
    total_road, total_back, total_veg = len(road_data), len(back_data), len(veg_data)
    print "shapes",road_data.shape, back_data.shape, veg_data.shape
    data = np.concatenate((road_data,back_data,veg_data))
    labels = np.concatenate((
        np.full((total_road,),0),
        np.full((total_back,),1),
        np.full((total_veg,),2),
    ))

    print "data.shape, labels.shape", data.shape, labels.shape
    print "\n0:road\n1:background\n2:vegetation\n3:muddy"
    print "total_road:", total_road
    print "total_background:", total_back
    print "total_veg:", total_veg
    return data, labels


# from utils import show_images
# from sklearn.utils import shuffle
#
# data, labels = load_patched_data(64,64)
# inds = shuffle(range(len(data)))[:10]
# show_images(data[inds],labels[inds])


def load_whole_image_data():
    from utils import onehot_images_dash
    from scipy.misc import imresize
    images = []
    # list = [0, 1. / 4, 1. / 2, 3. / 4]
    d = 2 #divisions
    list = np.arange(d)/d
    for img_file in glob.glob("data/uavdata/uav/*.jpg"):
        img = plt.imread(img_file)
        img = imresize(img,(1024,1024))
        n_r,n_c,n_ch = img.shape
        for i in list:
            for j in list:
                img_dash = img[int(i*n_r):int((i+1./d)*n_r),int(j*n_c):int((j+1./d)*n_c),:]
                images.append(img_dash)
    gts = []
    for img_file in glob.glob("data/uavdata/gts/*.jpg"):
        img = plt.imread(img_file)
        img = imresize(img, (1024, 1024))
        n_r, n_c, n_ch = img.shape
        for i in list:
            for j in list:
                img_dash = img[int(i*n_r):int((i+1./d)*n_r),int(j*n_c):int((j+1./d)*n_c),:]
                gts.append(img_dash)
    gts = onehot_images_dash(gts)

    return np.array(images),np.array(gts)


def load_bce_data():
    from utils import onehot_images_dash_dash
    from scipy.misc import imresize
    images = []

    d = 2. #divisions
    # list = [0, 1. / 4, 1. / 2, 3. / 4]
    list = np.arange(d)/d # points of cuts

    for img_file in glob.glob("data/uavdata/uav/*.jpg"): # Loop over all images in specified path
        img = plt.imread(img_file)
        img = imresize(img,(1024,1024))
        n_r,n_c,n_ch = img.shape
        for i in list:
            for j in list:
                img_dash = img[int(i*n_r):int((i+1./d)*n_r),int(j*n_c):int((j+1./d)*n_c),:]
                images.append(img_dash)
    gts = []
    for img_file in glob.glob("data/uavdata/gts/*.jpg"):
        img = plt.imread(img_file)
        img = imresize(img, (1024, 1024))
        n_r, n_c, n_ch = img.shape
        for i in list:
            for j in list:
                img_dash = img[int(i*n_r):int((i+1./d)*n_r),int(j*n_c):int((j+1./d)*n_c),:]
                gts.append(img_dash)

    gts = onehot_images_dash_dash(gts) # Transform Ground Truths to Binary distribution

    return np.array(images),np.array(gts)

