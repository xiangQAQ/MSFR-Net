import numpy as np
import os
import SimpleITK as sitk
from evalution_metrics import *
import pandas as pd
from pandas import Series, DataFrame

IDlist=[]
for root, dirs, files in os.walk('result/'):
    for dir in dirs:
        IDlist.append(dir)

IDlist = [i.split('.')[0][0:-5] for i in files]
print(IDlist[6], IDlist[7],IDlist[12],IDlist[15],IDlist[22],IDlist[25],IDlist[26])
me_l = ['DC_whole','DC_en','DC_core', 'PPV_whole','PPV_en','PPV_core', 'Sen_whole','Sen_en','Sen_core', 'Sep_whole','Sep_en','Sep_core', 'HD_whole','HD_en','HD_core']
table = pd.DataFrame(index=IDlist,columns=me_l)
for t in IDlist:
    gt = sitk.ReadImage(os.path.join('dataset/brats2018/test/HGG/'+ t +'/'+ t +'_seg.nii.gz'))
    gt = sitk.GetArrayFromImage(gt)[None].astype(np.float32)

    predicted_images = sitk.ReadImage('result/'+ t + '_pred.nii')
    predicted_images = sitk.GetArrayFromImage(predicted_images)[None].astype(np.float32)

    predicted_images[predicted_images==3]=4
    gt[gt==3]=4
    predicted_images = np.squeeze(predicted_images, 0)
    gt = np.squeeze(gt, 0)

    # compute the evaluation metrics
    Dice_complete = DSC_whole(predicted_images, gt)
    Dice_enhancing = DSC_en(predicted_images, gt)
    Dice_core = DSC_core(predicted_images, gt)

    PPV_complete = ppv_whole(predicted_images, gt)
    PPV_enhancing = ppv_en(predicted_images, gt)
    PPV_core = ppv_core(predicted_images, gt)

    Sensitivity_whole = sensitivity_whole(predicted_images, gt)
    Sensitivity_en = sensitivity_en(predicted_images, gt)
    Sensitivity_core = sensitivity_core(predicted_images, gt)

    Specificity_whole = specificity_whole(predicted_images, gt)
    Specificity_en = specificity_en(predicted_images, gt)
    Specificity_core = specificity_core(predicted_images, gt)

    Hausdorff_whole = hausdorff_whole(predicted_images, gt)
    Hausdorff_en = hausdorff_en(predicted_images, gt)
    Hausdorff_core = hausdorff_core(predicted_images, gt)

    table.loc[t] = [Dice_complete, Dice_enhancing, Dice_core, PPV_complete, PPV_enhancing, PPV_core,
                    Sensitivity_whole, Sensitivity_en, Sensitivity_core, Specificity_whole,
                    Specificity_en, Specificity_core, Hausdorff_whole, Hausdorff_en, Hausdorff_core]

mean = table.mean()
table = table.append(mean, ignore_index=True)
table.index = Series((IDlist+['mean']))
table.to_csv('pred_me.csv', index=False)
xx=1

