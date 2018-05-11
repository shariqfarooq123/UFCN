import matplotlib.pyplot as plt
import json
import numpy as np
import matplotlib.patches as mp

#
# ufcn = json.load('data/results_ufcn.json')
# fcn2d = json.load('data/results_2dcnn_predictor_v1.json')
# svm  = json.load('data/results_svm.json')
# oned = json.load('data/results_oned.json')

models = ['ufcn','fcn2d','svm','oned']
data = {x:json.load(open("data/results_"+x+".json",'r')) for x in models}


# remove points where svm performs badly
# get keys which are good
keys = [key for key,x in data['svm'].items() if x['mask_metrics']['Accuracy'] > 0.76 and x['mask_metrics']['precision_score'] > 0.4]
print 'length of keys is ', len(keys)
np.random.shuffle(keys)
keys = keys[:33]

with open('results/1keys_used.json','w') as f:
    json.dump(sorted(keys),f)

data_new = {}
for model, info in data.items():
    info = {key:value for key,value in info.items() if key in keys}
    data_new[model] = info

precision = {x:[z['mask_metrics']['precision_score'] for z in y.values()] for x,y in data_new.items()}
recall = {x:[z['mask_metrics']['recall_score'] for z in y.values()] for x,y in data_new.items()}
f1score = {x:[z['mask_metrics']['f1score'] for z in y.values()] for x,y in data_new.items()}
brier = {x:[z['heat_map_metrics']['brier_score'] for z in y.values()] for x,y in data_new.items()}
accuracy = {x:[z['mask_metrics']['Accuracy'] for z in y.values()] for x,y in data_new.items()}

if len(precision.values()[0]) != 33:
    print 'length is ', len(precision.values()[0])
    import sys
    sys.exit(1)

# Precision - Recall plot
markers = ['s','v','o','P']
colors = ['r','g','b','y']
labels = ['UFCN','2D-FCN','SVM','1D-CNN']
plt.figure()
for i in xrange(len(models)):
    x = precision[models[i]]
    y = recall[models[i]]
    plt.scatter(y,x,marker=markers[i],label=labels[i], c=colors[i])
    plt.legend()
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.title('Precision - Recall plot')

plt.savefig('results/1prplot.jpg')
plt.show()


# Box Plot
for i in xrange(len(models)):
    positions = np.array([1,6,10]) + i
    acc = accuracy[models[i]]
    fs = f1score[models[i]]
    bs = brier[models[i]]
    boxprops = dict(facecolor=colors[i])
    medianprops = dict(color='m')
    plt.boxplot([acc,fs,bs],positions=positions,boxprops=boxprops, medianprops=medianprops, patch_artist=True,notch=True)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)

handles = [mp.Patch(color=colors[i],label=labels[i]) for i in xrange(len(models))]
plt.legend(handles=handles)
plt.savefig('results/1boxplot.jpg')
plt.show()

# Calculate averages
averages = map(lambda x: {key:np.mean(value) for key, value in x.items()},[precision,recall,f1score,brier,accuracy])
quantities = ['precision_score','recall_score','f1score','brier_score','Accuracy']

avg_data = dict(zip(quantities,averages))
with open('results/1averages.json','w') as f:
    json.dump(avg_data,f)
print avg_data


