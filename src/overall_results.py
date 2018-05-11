import json
import  numpy as np


models = ['ufcn','fcn2d','svm','oned']
data = {x:json.load(open("data/results_"+x+".json",'r')) for x in models}

with open('results/1keys_used.json','r') as f:
    keys = json.load(f)

data_new = {}
for model, info in data.items():
    info = {key:value for key,value in info.items() if key in keys}
    data_new[model] = info

confusion_matrices = {x:[z['mask_metrics']['confusion_matrix'] for z in y.values()] for x,y in data_new.items()}
briers = {x:[z['heat_map_metrics']['brier_score'] for z in y.values()] for x,y in data_new.items()}


overall_confusion_matrices = {model:np.sum(conf_mats,axis=0).tolist() for model,conf_mats in confusion_matrices.items()}
print overall_confusion_matrices

with open('results/1overall_confusion_matrices.json','w') as f:
    json.dump( overall_confusion_matrices,f)

'''    Thus in binary classification, the count of true negatives is
    :math:`C_{0,0}`, false negatives is :math:`C_{1,0}`, true positives is
    :math:`C_{1,1}` and false positives is :math:`C_{0,1}`.
'''

def calc_stats(conf_mat):
    tn = np.array(conf_mat)[ 0, 0]
    fn = np.array(conf_mat)[ 1, 0]
    tp = np.array(conf_mat)[ 1, 1]
    fp = np.array(conf_mat)[ 0, 1]
    return tn,fn,tp,fp

def precision(conf_mat):
    tn,fn,tp,fp = calc_stats(conf_mat)
    return np.divide(tp,tp+fp,dtype='float')

def recall(conf_mat):
    tn,fn,tp,fp = calc_stats(conf_mat)
    return np.divide(tp,tp+fn,dtype='float')

def fscore(conf_mat):
    p = precision(conf_mat)
    r = recall(conf_mat)
    return 2*p*r/(p + r)

def accuracy(conf_mat):
    tn,fn,tp,fp = calc_stats(conf_mat)
    return np.divide(tp+tn,tp+fp+tn+fn,dtype='float')


overall_brier = {model:np.mean(b) for model,b in briers.items()}



overall_metrics = {
    'precision_score':{model:precision(conf_mat) for model,conf_mat in overall_confusion_matrices.items()},
    'recall_score':{model:recall(conf_mat) for model,conf_mat in overall_confusion_matrices.items()},
    'f1score':{model:fscore(conf_mat) for model,conf_mat in overall_confusion_matrices.items()},
    'Accuracy':{model:accuracy(conf_mat) for model,conf_mat in overall_confusion_matrices.items()},
    'brier_score':overall_brier
}

print 'overall metrics are',overall_metrics

with open('results/1overall_metrics.json','w') as f:
    json.dump(overall_metrics,f)


