"""
Run this file to automatically save all results of a given model, including compared images of heat map and threshold, tables of metrics etc
"""


from utils import mkdir_p,get_complete_result
import matplotlib.pyplot as plt
import sys
import cPickle


heat_map_dir = "results/final/heat_maps_compared"
mask_dir = "results/final/masks_compared"
table_dir = "results/final/tables"
to_continue = mkdir_p(heat_map_dir)
if not to_continue:
    print "Overwrite declined, program terminated "
    sys.exit(0)

to_continue = mkdir_p(mask_dir)
if not to_continue:
    print "Overwrite declined, program terminated "
    sys.exit(0)

to_continue = mkdir_p(table_dir)
if not to_continue:
    print "Overwrite declined, program terminated "
    sys.exit(0)


selected_image_numbers = ['04','07','09','11','15']

results = []
for img_number in selected_image_numbers:
    img_path = "data/uavdata/uav/uav ("+img_number+").jpg"
    gt_path = "data/uavdata/gts/uav"+img_number+"gt.jpg"
    heat_path = heat_map_dir+"/heat"+img_number+".jpg"
    mask_path = mask_dir+"/mask"+img_number+".jpg"
    table_path = table_dir+"table"+img_number+".jpg"

    input_image = plt.imread(img_path)
    input_gt = plt.imread(gt_path)

    complete_result = get_complete_result(input_image,input_gt,img_number)
    complete_result.get_hfig().savefig(heat_path)
    complete_result.get_mfig().savefig(mask_path)
    complete_result.get_table().savefig(table_path)
    results.append(complete_result)

with open("final_results1.pkl","wb") as file:
    cPickle.dump(results,file,cPickle.HIGHEST_PROTOCOL)

