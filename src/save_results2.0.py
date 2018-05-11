"""
Smarter version
"""

import sys

from loader import load_bce_data
from keras.models import load_model
from custom_functions import dice_coef_loss, dice_coef
from helper import Result
from utils import separate_channels,combine_channels,prepare_result,mkdir_p,stitch
from tqdm import tqdm
import cPickle
import json

model_name = raw_input("Enter model name:")
result_dir = "results/tr_"+model_name
model = load_model("models/"+model_name+".h5",custom_objects={"dice_coef":dice_coef,"dice_coef_loss":dice_coef_loss})
to_continue = mkdir_p(result_dir)
if not to_continue:
    print("Overwrite declined, program interrupted!")
    sys.exit(0)

print "loading data.."
data, labels  = load_bce_data()
print "done"
# rand_inds = shuffle(range(len(data)))
# data = data[rand_inds]
# labels = labels[rand_inds]
data = separate_channels(data)
print "predicting.."
preds = model.predict(data,batch_size=1,verbose=1)
print "done"
data = combine_channels(data)
predicted_images = preds.reshape((-1,data[0].shape[0],data[0].shape[1]))

#data = stitch(data,1024,4)
#predicted_images = stitch(predicted_images,1024,4)
#labels = stitch(labels,1024,4)


print "saving results to "+result_dir
count = 0
results = {}
for orig_image, label_image, heat_map in tqdm(zip(data,labels,predicted_images)):
    count = count +1
    result = prepare_result(orig_image,label_image,heat_map,heat_map_metrics=[Result.BRIER_SCORE],mask_metrics=[Result.CONFUSION_MATRIX,Result.F1_SCORE,Result.ACCURACY, Result.PRECISION_SCORE, Result.RECALL_SCORE])
    result.show()
    result.save(result_dir+"/tr_"+model_name+"_"+str(count)+".jpg")
    mask_metrics = result.mask_metrics
    mask_metrics[Result.CONFUSION_MATRIX] = mask_metrics[Result.CONFUSION_MATRIX].tolist()
    r = dict(heat_map_metrics=result.heat_map_metrics, mask_metrics=mask_metrics)
    results[count] = r
    # result.print_results()
    #sys.stdout.write("#")



with open('data/results_'+model_name+'.json', 'w') as f:
    print 'saving results...'
    json.dump(results,f)
    print 'saved!'
print "\n:)"


