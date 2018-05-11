"""
This file contains some helper classes used to make code more organized
"""
import matplotlib.pyplot as plt
from keras.models import load_model
from skimage import color
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score,log_loss,brier_score_loss,precision_score,recall_score
from sklearn.metrics import confusion_matrix
from utils import divide_image,separate_channels,combine_channels,stitch
from scipy.misc import imresize
from utilsr import image_builder, inverse_onehot
import numpy as np
from utils import prepare_result
from custom_functions import dice_coef,dice_coef_loss

class Result():
    """
        list_images in order : orignal, heat_map, mask, gt
    """



    KL_DIVERGENCE = "KL_Divergence"
    ACCURACY = "Accuracy"
    F1_SCORE = "f1score"
    CONFUSION_MATRIX = "confusion_matrix"
    BRIER_SCORE = "brier_score"
    PRECISION_SCORE = "precision_score"
    RECALL_SCORE = "recall_score"


    def __init__(self,orig_image,heat_map,mask,gt,**kwargs):
        from pylab import rcParams
        rcParams['figure.figsize'] = 25, 25
        self.original_image = orig_image
        self.heat_map = heat_map
        self.mask = mask
        self.gt = gt
        self.list_heat_map_metrics = kwargs.get("heat_map_metrics")
        self.list_mask_metrics = kwargs.get("mask_metrics")
        self._build_result()


    def _build_result(self):

        self.heat_map_metrics = self._calculate_heat_metrics()
        self.mask_metrics = self._calculate_mask_metrics()
        return


    def show(self):
        plt.close()
        font = {
            "color": "white",
            "size": 12,
            "weight": "normal"
        }
        self.fig = plt.figure()
        plt.subplot('221')
        plt.title("Input Image")
        plt.imshow(self.original_image)
        plt.axis('off')

        plt.subplot('222')
        plt.title("Ground Truth")
        plt.imshow(self.gt)
        plt.axis('off')

        plt.subplot('223')
        plt.title("Heat Map of Road Probability")
        plt.imshow(self.heat_map)

        if self.heat_map_metrics is not None:
            info = ""
            for key,value in self.heat_map_metrics.iteritems():
                info = info +key+":"+str(round(value,3))+"\n"

            plt.text(self.heat_map.shape[1]-60,36,info,fontdict=font,ha='right',va='top')
        plt.axis('off')

        plt.subplot('224')
        plt.title("Mask")
        plt.imshow(self.mask)
        plt.axis('off')

        if self.mask_metrics is not None:
            info = ""
            for key, value in self.mask_metrics.iteritems():
                info = info + key + ":" + str(value) + "\n"

            plt.text(self.mask.shape[1]-60, 36, info, fontdict=font,ha='right',va='top')

        plt.show()
        plt.close()


    def save(self,filename):
        plt.savefig(filename)
        plt.close()

    def _calculate_heat_metrics(self):
        if self.list_heat_map_metrics is not None:
            heat_map_metrics = {}
            if self.KL_DIVERGENCE in self.list_heat_map_metrics:
                heat_map_metrics[self.KL_DIVERGENCE] = round(log_loss(self.gt.ravel(),self.heat_map.ravel()),5)

            if self.BRIER_SCORE in self.list_heat_map_metrics:
                print "woo",self.gt.shape, self.heat_map.shape
                heat_map_metrics[self.BRIER_SCORE] = round(brier_score_loss(self.gt.ravel(),self.heat_map.ravel()),5)

            return heat_map_metrics
        else:
            return None


    def _calculate_mask_metrics(self):
        if self.list_mask_metrics is not None:
            mask_metrics = {}
            if self.ACCURACY in self.list_mask_metrics:
                mask_metrics[self.ACCURACY] = round(accuracy_score(self.gt.ravel(),self.mask.ravel()),5)

            if self.F1_SCORE in self.list_mask_metrics:
               mask_metrics[self.F1_SCORE] = round(f1_score(self.gt.ravel(),self.mask.ravel()),5)

            if self.CONFUSION_MATRIX in self.list_mask_metrics:
                mask_metrics[self.CONFUSION_MATRIX] = round(confusion_matrix(self.gt.ravel(),self.mask.ravel()),5)

            if self.PRECISION_SCORE in self.list_mask_metrics:
                            mask_metrics[self.PRECISION_SCORE] = round(precision_score(self.gt.ravel(),self.mask.ravel()),5)

            if self.RECALL_SCORE in self.list_mask_metrics:
                mask_metrics[self.RECALL_SCORE] = round(recall_score(self.gt.ravel(), self.mask.ravel()),5)

            return mask_metrics
        else:
            return None

    def __str__(self):
        return str(self.heat_map_metrics)+"\n"+str(self.mask_metrics)

    def print_results(self):
        print "\nHeat map metrics:"
        info = ""
        for key, value in self.heat_map_metrics.iteritems():
            info = info + key + ": " + str(value) + "\n"
        print info
        print "\nMask metrics:"
        info = ""
        for key, value in self.mask_metrics.iteritems():
            info = info + key + ": " + str(value) + "\n"
        print info
        print "-"*50

    def get_metric_value(self,metric_name):
        h_metric = self.heat_map_metrics.get(metric_name)
        if h_metric is not None:
            return h_metric
        m_metric = self.mask_metrics.get(metric_name)
        if metric_name is not None:
            return m_metric

        raise TypeError("Invalid argument 'metric_name', Make sure result was build with this metric included.")

class Tester:
    def __init__(self,model,test_image):
        self.model = model
        self.test_image = imresize(test_image,(1024,1024))


    def unet_predict(self):
        d = divide_image(test_image=self.test_image)
        data = separate_channels(d)
        preds = self.model.predict(data,verbose=1)
        data = combine_channels(data)
        predicted_image = preds.reshape((-1,data[0].shape[0],data[0].shape[1]))
        predicted_image = stitch(predicted_image,1024,4)[0]
        print "heatmap of unet",predicted_image.shape
        return predicted_image

    def oned_predict(self):
        model = load_model("models/oned.h5")
        input_image = self.test_image

        image1 = input_image.astype('float32')
        image1 = image1 / 255.0
        image1_hsv = color.rgb2hsv(input_image)

        test = np.reshape(image1, (-1, 3))
        test2 = np.reshape(image1_hsv, (-1, 3))
        mem_new = np.append(test, test2, axis=1)
        mem_new = np.append(mem_new, test, axis=1)
        mem_new = np.reshape(mem_new, (-1, 1, 9))

        pred_nothot = model.predict(mem_new, verbose=1)
        pred = np.array(inverse_onehot(pred_nothot)).reshape((image1.shape[0],image1.shape[1]))
        mask = pred

        labels = []
        for row in pred_nothot:
            labels.append(row[1])
        heat_map = np.reshape(labels, (image1.shape[0], image1.shape[1]))
        print "heatmap of oned",heat_map.shape

        return heat_map

    def set_test_image(self,img):
        self.test_image = imresize(img,(1024,1024))




class UFCN:
    def __init__(self):
        self.model = load_ufcn()
        self.func = None

    def preprocess(self,test_image):
        if isinstance(test_image,str):
            test_image = plt.imread(test_image)

        test_image = imresize(test_image,(1024,1024))
        d = divide_image(test_image=test_image)
        print "d.shape is ",np.shape(d)
        data = separate_channels(d)
        self.orig_shape = (-1,d[0].shape[0],d[0].shape[1])
        return data

    def postprocess(self,preds):
        predicted_image = preds.reshape(self.orig_shape)
        predicted_image = stitch(predicted_image, 1024, 4)[0]
        return predicted_image

    def predict(self,test_image):
        data = self.preprocess(test_image)
        print "data.shape is ",data.shape
        preds = self.model.predict(data,verbose=1)
        predicted_image = self.postprocess(preds)
        print "heatmap of unet",predicted_image.shape
        return predicted_image

    def take_out(self,layer_name,image,average=True):
        if isinstance(image, str):
            image = plt.imread(image)

        images = self.preprocess(image)
        if not self.func:
            self.func = self.make_func(layer_name)
        feature_maps = self.func([images])[0]
        if average:
            feature_maps = np.mean(feature_maps,axis=1)
        else:
            feature_maps = combine_channels(feature_maps)
        feature_map = self.custom_stitch(feature_maps)

        if not average:
            feature_map = separate_channels(np.array([feature_map]))[0]
        return feature_map

    def custom_stitch(self,data):
        assert len(data) == 4
        return np.vstack((np.hstack((data[0],data[1])),np.hstack((data[2],data[3]))))
        #orig_cols = 2 * data[0].shape[0]
        #return stitch(data,orig_cols,4)


    def load_ufcn(self):
        #from custom_functions import dice_coef
        from keras.models import load_model
        #ufcn = load_model('models/final.h5', custom_objects={'dice_coef': dice_coef})
        ufcn = load_model('models/ufcn.h5')

        # Make sure it's dunet7
        #ufcn.load_weights('models/dunet7_wts.h5')
        ufcn.trainable = False

        return ufcn

    def make_func(self,layer_name, model=None):
        if not model:
            model = load_ufcn()
        from keras import backend as K
        func = K.function([model.input], [model.get_layer(layer_name).output])
        return func





class CompleteResult:
    def __init__(self,result_name):
        self.name = result_name

    def set_hfig(self,figure):
        self.hfig = figure

    def set_mfig(self,figure):
        self.mfig = figure

    def set_table(self,table_figure):
        self.table_fig = table_figure

    def set_metrics(self,metrics_dict):
        self.metrics = metrics_dict

    def get_hfig(self):
        return self.hfig

    def get_mfig(self):
        return self.mfig

    def get_table(self):
        return self.table_fig

    def get_metrics(self):
        return self.metrics


class ResultBuilder:
    def __init__(self,input_image,input_gt):
        from utils import onehot_images_dash_dash
        input_gt = onehot_images_dash_dash([input_gt])[0]
        fcn_model = load_model("models/dunet7.h5",custom_objects={"dice_coef":dice_coef,"dice_coef_loss":dice_coef_loss})
        tester = Tester(fcn_model,input_image)
        fcn_heat_map = tester.unet_predict()
        oned_heat_map = tester.oned_predict()

        self.result_fcn = prepare_result(input_image,input_gt,fcn_heat_map,heat_map_metrics=[Result.BRIER_SCORE],mask_metrics=[Result.ACCURACY,Result.F1_SCORE,
                                                                                                                          Result.PRECISION_SCORE,
                                                                                                                          Result.RECALL_SCORE])
        self.result_oned = prepare_result(input_image, input_gt, oned_heat_map, heat_map_metrics=[Result.BRIER_SCORE],
                                    mask_metrics=[Result.ACCURACY, Result.F1_SCORE,
                                                  Result.PRECISION_SCORE,
                                                  Result.RECALL_SCORE])

    def build_hfig(self):
        plt.close()
        fig = plt.figure()
        plt.subplot('141')
        plt.imshow(self.result_fcn.original_image)
        plt.subplot('142')
        plt.imshow(self.result_fcn.gt)
        plt.subplot('143')
        plt.imshow(self.result_oned.heat_map)
        plt.subplot('144')
        plt.imshow(self.result_fcn.heat_map)
        # fig.show()
        fig.tight_layout()
        return fig

    def build_mfig(self):
        plt.close()
        fig = plt.figure()
        plt.subplot('141')
        plt.imshow(self.result_fcn.original_image)
        plt.subplot('142')
        plt.imshow(self.result_fcn.gt)
        plt.subplot('143')
        plt.imshow(self.result_oned.mask)
        plt.subplot('144')
        plt.imshow(self.result_fcn.mask)
        # fig.show()
        fig.tight_layout()
        return fig

    def build_table(self):
        plt.close()
        fig = plt.figure()
        # table_values = np.zeros((2,5))
        # table_values = []
        a = self.result_fcn.mask_metrics
        b = self.result_fcn.heat_map_metrics
        print a.values()
        print b.values()


        fcn_metrics = np.array(self.result_fcn.mask_metrics.values()+(self.result_fcn.heat_map_metrics.values()))
        print fcn_metrics
        oned_metrics = np.array(self.result_oned.mask_metrics.values()+(self.result_oned.heat_map_metrics.values()))
        table_values = np.vstack((fcn_metrics,oned_metrics))
        col_labels = (self.result_fcn.mask_metrics.keys()+(self.result_fcn.heat_map_metrics.keys()))

        print table_values

        row_labels = ('U-fcn','1D-conv')
        plt.axis('off')
        t = plt.table(colLabels=col_labels,
                  rowLabels=row_labels,
                  cellText=table_values,
                  loc='center')
        t.set_fontsize(26)
        t.scale(1,3)

        return fig

    def build_metrics_dict(self):
        metrics = {
            "heat_map_metrics_fcn":self.result_fcn.heat_map_metrics,
            "heat_map_metrics_oned":self.result_oned.heat_map_metrics,
            "mask_metrics_fcn":self.result_fcn.mask_metrics,
            "mask_metrics_oned":self.result_oned.mask_metrics,
        }

        return metrics



def load_ufcn():
    from custom_functions import dice_coef
    from keras.models import load_model

    ufcn = load_model('models/final.h5',custom_objects={'dice_coef':dice_coef})

    # Make sure it's dunet7
    ufcn.load_weights('models/dunet7_wts.h5')
    return ufcn


def make_func(layer_name, model= None):
    if not model:
        model = load_ufcn()
    from keras import backend as K
    func = K.function([model.input],[model.get_layer(layer_name).output])
    return func

def func_predict(func,image):
    from utils import convert_ordering_tf_to_th
    from scipy.misc import imresize
    image = imresize(image,(512,512))
    image = convert_ordering_tf_to_th(np.array([image]))[0]
    out = func([[image]])[0][0]
    avg = np.mean(out,axis=0)
    return avg

def take_out(layer_name,image):
    if isinstance(image,str):
        import matplotlib.pyplot as plt
        image = plt.imread(image)

    f = make_func(layer_name)
    out = func_predict(f,image)
    return out
