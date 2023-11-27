"""
By Kibrom Berihu Girum, LITO, insititute curie, France. 
kibrom2bATgmail.com 
"""
# import libraries 
import os, glob 
from pathlib import Path

import numpy as np 
import pandas as pd
from collections import defaultdict


# library for dmax computation
from skimage.measure import label, regionprops
from skimage import data, util
from scipy.spatial import distance


import math 
import skimage 
from scipy.stats import spearmanr

import nibabel as nib 
import nilearn
import scipy.ndimage as scim

# library for plotting, progress and time 
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
from matplotlib.pyplot import *
from tqdm import tqdm 
import time 


"""

The distance between the spleen and the farthest lesion from the spleen is named hereafter DLS (distant lesion from the spleen)
The standard deviation of the distances measured between the spleen and other lesions is named hereafter LS (lesion spread)

"""



def get_features(target_organ, mask, res_pet):
    """ function to get the x, y, z coordinates of lesion (mask) and target organ (mask of target organ)
    target_organ: the 3D mask (binary image) of the target organ, target organ areas 1 and others 0
    mask: 3D binary image (mask) of the lesion areas, lesion areas 1 and others 0.
    res_pet: the pixel dimensions of the 3D images 
    """

    # label the lesion areas of the 3D image     
    mask = util.img_as_ubyte(mask) > 0
    mask_labeled = label(mask, connectivity=mask.ndim)
    props = regionprops(mask_labeled)

    # features 
    features = defaultdict(list)
    df = pd.DataFrame()

    # if there is no lesion (labeled 1) return 0, healthy 
    if mask_labeled.max() ==0:
#         features = dict.fromkeys(features, 0)
        features['label'].append(float(math.nan))
        features['xcenter'].append(float(math.nan))
        features['ycenter'].append(float(math.nan))
        features['zcenter'].append(float(math.nan))
        features['area'].append(float(math.nan))
        features['TMTV'].append(float(math.nan))
        features['Ntumors'].append(float(math.nan))
    else:
        
        # calculate features for each lesion 
        for label_id in range(len(props)):        
    #        mask_at_given_label = (mask_labeled == label_id).astype(dtype=np.uint8)
    #         x, y, z = np.nonzero(mask_at_given_label)
    #         features['xcord'] = list(x)
    #         features['ycord']= list(y)
    #         features['zcord']= list(z) 
            features['label'].append(label_id)
            features['xcenter'].append(round(np.array(props[label_id].centroid)[0], 3))
            features['ycenter'].append(round(np.array(props[label_id].centroid)[1], 3))
            features['zcenter'].append(round(np.array(props[label_id].centroid)[2], 3))
            features['area'].append(round(props[label_id].area, 3))
        
        TMTV = np.sum(mask>0)*res_pet[0]*res_pet[1]*res_pet[2]
        for i in range(len(features['xcenter'])):
            features['TMTV'].append(TMTV)
            features['Ntumors'].append(len(props))

    df_update = pd.DataFrame(data=features)
    df = pd.concat([df, df_update], axis=0)
    
    
    features = defaultdict(list)
    target_organ_label = (target_organ > 0).astype(dtype=np.uint8)
    

    target_organ_label_label = util.img_as_ubyte(target_organ_label) > 0        
    target_organ_label_label = label(target_organ_label_label, connectivity=target_organ_label_label.ndim)
    props = regionprops(target_organ_label_label)

    if np.sum(target_organ_label) == 0:        
        features['xcenter_target_centeriod'].append(float(math.nan))
        features['ycenter_target_centeriod'].append(float(math.nan))
        features['zcenter_target_centeriod'].append(float(math.nan))

        features['xcenter_target'].append(float(math.nan))
        features['ycenter_target'].append(float(math.nan))
        features['zcenter_target'].append(float(math.nan))
 
    else:
        count = (target_organ_label == 1).sum()
        x_center, y_center, z_center = np.argwhere(target_organ_label == 1).sum(0)/count
        for label_id in range(len(props)): 
            features['xcenter_target_centeriod'].append(round(np.array(props[label_id].centroid)[0], 3))
            features['ycenter_target_centeriod'].append(round(np.array(props[label_id].centroid)[1], 3))
            features['zcenter_target_centeriod'].append(round(np.array(props[label_id].centroid)[2], 3))
            features['center_area'].append(round(props[label_id].area, 3))

            features['xcenter_target'].append(round(x_center, 3))
            features['ycenter_target'].append(round(y_center, 3))
            features['zcenter_target'].append(round(z_center, 3))


        for index in range(len(df)-len(props)):
            features['xcenter_target_centeriod'].append(float(math.nan))
            features['ycenter_target_centeriod'].append(float(math.nan))
            features['zcenter_target_centeriod'].append(float(math.nan))


            features['xcenter_target'].append(float(math.nan))
            features['ycenter_target'].append(float(math.nan))
            features['zcenter_target'].append(float(math.nan))
            features['center_area'].append(float(math.nan))            

    #         plt.imshow(np.amax(target_organ, axis=2))
    #         plt.scatter (np.median(y), np.median(x)) # np.mean(y),
    #         plt.scatter (centers[1], centers[0]) # np.mean(y),
    #         plt.show()

        
        
        
#     target_organ_label = util.img_as_ubyte(target_organ) > 0        
#     target_organ_label = label(target_organ_label, connectivity=target_organ_label.ndim)
#     props = regionprops(target_organ_label)
   
#     if target_organ_label.max() ==0:
#         features['xcenter_target_centeriod'].append(float(math.nan))
#         features['ycenter_target_centeriod'].append(float(math.nan))
#         features['zcenter_target_centeriod'].append(float(math.nan))

#         features['xcenter_target'].append(float(math.nan))
#         features['ycenter_target'].append(float(math.nan))
#         features['zcenter_target'].append(float(math.nan))
#     else:
# #         taget_label = (target_organ >0).astype(dtype=np.uint8)   
# #         x, y, z = np.nonzero(taget_label)   
# #         centers = np.array(props[0].centroid)          
        
#         # calculate features, observed more than one region of spleen 
#         for label_id in range(len(props)):   
#             mask_at_given_label = (mask_labeled == label_id).astype(dtype=np.uint8)
#             mask_at_given_label [mask_at_given_label >0] = 1
            
#             mask_amax = np.amax(mask_at_given_label, axis=2)
#             mask_amax = np.array(mask_amax)
#             count = (mask_amax == 1).sum()
#             x_center, y_center = np.argwhere(mask_amax == 1).sum(0)/count
            
#             mask_amax = np.amax(mask_at_given_label, axis=1)
#             mask_amax = np.array(mask_amax)
#             count = (mask_amax == 1).sum()
#             _, z_center = np.argwhere(mask_amax == 1).sum(0)/count
            
            
            
# #             mask_at_given_label = np.array(mask_at_given_label)
# #             count = (mask_at_given_label ==1).sum()        
# #             x_center, y_center, z_center = np.argwhere(mask_at_given_label ==1).sum(0)/count
        
#             features['xcenter_target_centeriod'].append(round(np.array(props[label_id].centroid)[0], 3))
#             features['ycenter_target_centeriod'].append(round(np.array(props[label_id].centroid)[1], 3))
#             features['zcenter_target_centeriod'].append(round(np.array(props[label_id].centroid)[2], 3))
#             features['center_area'].append(round(props[label_id].area, 3))
            
#             features['xcenter_target'].append(round(x_center, 3))
#             features['ycenter_target'].append(round(y_center, 3))
#             features['zcenter_target'].append(round(z_center, 3))
            
            
#         for index in range(len(df)-len(props)):
#             features['xcenter_target_centeriod'].append(float(math.nan))
#             features['ycenter_target_centeriod'].append(float(math.nan))
#             features['zcenter_target_centeriod'].append(float(math.nan))

            
#             features['xcenter_target'].append(float(math.nan))
#             features['ycenter_target'].append(float(math.nan))
#             features['zcenter_target'].append(float(math.nan))
#             features['center_area'].append(float(math.nan))            
            
# #         plt.imshow(np.amax(target_organ, axis=2))
# #         plt.scatter (np.median(y), np.median(x)) # np.mean(y),
# #         plt.scatter (centers[1], centers[0]) # np.mean(y),
# #         plt.show()

    df_update = pd.DataFrame(data=features)
    df = pd.concat([df, df_update],axis=1)    
    return df 

def superimpose_segmentation_images(pet_gt_prd_display, file_name,
                                    logzscore=None, xyz=None):
    """
    Args:
        pet_ct_gt_prd:
        file_name:
        logzscore:
    """
    pet, gt, prd = pet_gt_prd_display[0], pet_gt_prd_display[1], pet_gt_prd_display[2]
    '''
    if logzscore == "log":
        pet = np.log(pet + 1)
    elif logzscore == "zscore":
        pet = (pet - np.mean(pet)) / (np.std(pet) + 1e-8)
    elif logzscore == "clipping":
        pet[pet > 50] = 50
        pet /= 50
    else:
        pet = np.log(pet + 1)
    '''

    pet[pet>10] = 10

    pet[pet<0] = 0
    
    img = pet
    try:
        img = np.squeeze(img, axis=-1)
    except:
        pass
    try:
        gt = np.squeeze(gt, axis=-1)
    except:
        pass

    try:
        prd = np.squeeze(prd, axis=-1)
    except:
        pass

    img = np.rot90(img)

    if len(prd):
        prd = np.rot90(prd)
        prd[prd > 0] = 1

    img = 10 - img
    if len(gt):
        gt = np.rot90(gt)
        gt[gt > 0] = 1

        # miss classified regions
        prd_error = prd + gt
        prd_error[prd_error != 1] = 0
        dice = 0
    else:
        dice = 'unkown'

    hfont = {'fontname': 'Arial'}
    fontsize_ = 12
    # for clr in color:
    print("\n Image ID: \t %s", str(file_name))
    fig, axs = plt.subplots(1, 3, figsize=(10, 10))
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title('PET image', **hfont, fontsize=fontsize_)
    axs[0].set_xticklabels([])
    axs[0].set_yticklabels([])


    axs[1].imshow(img, cmap='gray')
    gt_copy = gt.copy()
    gt_copy = np.array(gt_copy)
    count = (gt_copy == 1).sum()
    x_center, y_center = np.argwhere(gt_copy==1).sum(0)/count
    
#         medain organ center [104.371  94.902 148.249], mean [112.55945763 102.34598305 156.08434915], and std [26.79571946 23.84247456 46.31650494]
#         xmedian, zmedian =104.371, 148.249

#         xmedian, zmedian = 96.895 , 142.556
# 
    
#         xmedian, zmedian =142.556, 96.895
#         96.895  89.109 142.556

    # medain organ center [ 94.3805  91.06   123.6565],
    # mean [105.79468182 101.1286     146.71554545],
    # and std [32.15215621 31.33882411 71.76159191]
    xmedian, zmedian = 94.3805, 123.6565

    x = np.arange(10)
    ys = [i+x+(i*x)**2 for i in range(10)]

    colors = cm.rainbow(np.linspace(0, 1, len(ys)))
#         print(xyz[0], xyz[1])
    '''
    print(x_center, y_center)
    xyz = xyz[::-1]
    '''
    


    if len(gt):
        gt = np.ma.masked_where(gt == 0, gt)
        viridis = cm.get_cmap('Set1')

        axs[1].imshow(gt, cmap=viridis)  # cmap='gray')#
        
        try:
            axs[1].scatter(y_center, x_center,
                    color='lime', marker="*", alpha=0.5)
            # axs[1].scatter(y_center, x_center,
            #         color=colors[5], alpha=0.5)
#                 axs[1].scatter(zmedian, xmedian,
#                         color=colors[-1], alpha=0.1)
        except:
            pass 
        
        axs[1].set_title('Spleen')
#             axs[1].set_title('Expert', **hfont, fontsize=fontsize_)
    else:
        axs[1].set_title('No ground truth provided', **hfont, fontsize=fontsize_)
    axs[1].set_xticklabels([])
    axs[1].set_yticklabels([])
    axs[1].set_aspect('equal')

    axs[2].imshow(img, cmap='gray')
    if len(prd):
        prd = np.ma.masked_where(prd==0, prd)
        viridis = cm.get_cmap('brg')
        axs[2].imshow(prd,  viridis)
        axs[2].set_title('Segmented lesion')
#             axs[2].set_title('CNN (Dice score: {dice}%)'.format(dice=dice), **hfont, fontsize=fontsize_)
    else:
        axs[2].set_title('predicted image not found'.format(dice=dice), **hfont, fontsize=fontsize_)
    
    
    try:
        axs[2].scatter(y_center, x_center,
                    color='lime', marker="*", alpha=0.5)
        # axs[2].scatter(zmedian, xmedian,
        #             color=colors[-1], alpha=0.1)
    except:
        pass 

    axs[2].set_xticklabels([])
    axs[2].set_yticklabels([])
    axs[2].set_aspect('equal')

    axs[0].axis('off')
    axs[1].axis('off')
    axs[2].axis('off')
    try:
        plt.savefig('../output/images/' + str(file_name) + '.png', dpi=300)
    except:
        [print("image not saved ")]
    plt.show()


# @ray.remote(num_cpus=os.cpu_count())
def worker(input_path, case_name, allmaskinonefolder=False):
    """
    :param input_path: Path to the directory that consists the folder for .nii files
    :return: read .nii, compute TMTV, Dmax, Dulk, DLS, and LS 
    """  
    calculated_features = defaultdict(list)
  
    if allmaskinonefolder:
        path_img_nii = str(input_path) + "*/" + str(
            case_name
            )  
    else:
        path_img_nii = str(input_path) + "/" + str(case_name)   
        files_grabbed = []
        for files in [".nii", ".nii.gz"]:
            files_grabbed.extend(glob.glob(path_img_nii + "/*/*" + files))        
    try:
        # mask 
        path_img_nii_mask = [ path for path in files_grabbed if "gt" in path.lower() and "gt_from_vienna" not in path.lower()] 
        # path_img_nii_mask = [ path for path in files_grabbed if "gt_from_vienna" in path.lower()] 

        print(f"considering mask from: {path_img_nii_mask}")
        path_img_nii_mask = path_img_nii_mask[0]
                
        # PET
        path_img_nii_pet = [ path for path in files_grabbed if "pet" in path.lower()]
        path_img_nii_pet = path_img_nii_pet[0]
        
        # path_target_organ = Path(os.path.join(path_img_nii, "organ_segmentation", "segmentations", "urinary_bladder.nii.gz"))
        # path_target_organ = Path(os.path.join(path_img_nii, "organ_segmentation", "segmentations", "heart_ventricle_left.nii.gz"))
        path_target_organ = Path(os.path.join(path_img_nii, "organ_segmentation", "segmentations", "spleen.nii.gz"))


                
        print(f" Path, Mask: {path_img_nii_mask} , pet: {path_img_nii_pet}, spleen: {path_target_organ} ")
    except:
        return None 

    # Read .nii files
    gt = nib.load(path_img_nii_mask)
    res_gt = gt.header.get_zooms()
 
    pet = nib.load(path_img_nii_pet)
    res_pet = pet.header.get_zooms()
    
    nospleenseg= defaultdict(list)
    nospleenseg = {"morethanoneregion": [], "oneregion":[], "nospleensegm":[]}
    
    try:
        target_organ = nib.load(path_target_organ)
        res_ct = target_organ.header.get_zooms()

        # register the lesion segmentation to the PET spacing
        gt = nilearn.image.resample_to_img(source_img=gt, target_img=pet,
                                           interpolation='nearest')
        
        # register the organ segmentation to the PET spacing 
        target_organ_aligned = nilearn.image.resample_to_img(source_img=target_organ, target_img=pet, interpolation='nearest')
        target_organ_aligned = np.asanyarray(target_organ_aligned.dataobj)
        
        pet_numpy = np.asanyarray(pet.dataobj)
        
        """
        If two or more than two regions are detected as part of the spleen segmentation apply connected compoenent analysis. 
        Consider the largest region (ideally) the spleen. Overal the spleen masks with PET and if they lay outside the 
        whole-body they are considered as outliers and delet them. After such visual check is necessary, for example in 
        one case of the REMARC data, there were difficulity in selecting the Spleen in such way. 
        """

        target_organ_aligned  = (target_organ_aligned > 0).astype(dtype=np.uint8)
        mask_labeled = skimage.measure.label(target_organ_aligned)

        if mask_labeled.max() > 1:
            nospleenseg['morethanoneregion'].append(case_name)    
        else:
            nospleenseg['oneregion'].append(case_name)
            
#         calculate 
        tmtv_initial = 0
        for selected_label in range(mask_labeled.max()):
            label = selected_label + 1

            mask_at_given_label = (mask_labeled == label).astype(dtype=np.uint8)
            
            # First check if the segemnted region is not outside the field of view 
            if not np.sum(pet_numpy * mask_at_given_label) < 0.5*np.sum(mask_at_given_label):
                tmtv = np.sum((mask_at_given_label > 0.5).astype(dtype=np.uint8) >0)
                if tmtv > tmtv_initial:
                    target_organ_aligned = mask_at_given_label.astype(dtype=np.uint8)
                    tmtv_initial = tmtv 

    except:
        print("segmentation of the target organ nor read !!")
        target_organ_aligned = np.zeros(np.asanyarray(pet.dataobj).shape[:3])
        target_organ = np.zeros((0, 0, 0))
        nospleenseg['nospleensegm'].append(case_name)
    
    # get images in array format 
    pet = np.asanyarray(pet.dataobj)
    gt = np.asanyarray(gt.dataobj)
    gt = (gt > 0).astype(dtype=np.uint8)

    try:
        gt, pet, target_organ_aligned = np.squeeze(gt), np.squeeze(pet), np.squeeze(target_organ_aligned)
    except:
        pass 

    # refineimages
    if len(gt.shape) == 4 and gt.shape[-1] ==2:
        gt = gt[..., 0]

    if len(pet.shape) == 4 and pet.shape[-1] ==2:
        pet = pet[..., 0]

    if pet.shape[0] != gt.shape[0]:
        gt = np.transpose(gt, (2, 0, 1))

    print(20*"=")
    print("pet shape: \t", pet.shape, " mask shape:\t", gt.shape, " orginal target organ shape:\t" , target_organ.shape,
         " target organ shape after aligned:\t", target_organ_aligned.shape, "resolution:\t", res_pet)
        
    if len(res_pet) ==4:
        res_pet = res_pet[:3]

    res_selected = res_pet #[4.0, 4.0, 4.0]    

    # get the image-based biomarkers
    df = get_features(target_organ_aligned.copy(), gt.copy(), res_selected)

    spleen_lesion_overlap = 0
    target_organ_aligned = (target_organ_aligned>0).astype(int)
    # target_organ_aligned = scim.binary_dilation(target_organ_aligned.copy(), iterations=3)
    if np.sum(target_organ_aligned*gt) > 0:
        spleen_lesion_overlap = 1

    # if spleen_lesion_overlap == 1:
    #display the coronal view with the spleen centeriod selected 
    xyz = [df['xcenter_target_centeriod'].values[0], df['zcenter_target_centeriod'].values[0]]
    pet_gt_prd_display = [ np.amax(pet, axis=1), np.amax(target_organ_aligned, axis=1), np.amax(gt, axis=1)]
    superimpose_segmentation_images(pet_gt_prd_display, 
                                    file_name=str(case_name) +'_1', xyz=xyz)

    # display the sagittal view with the spleen centeriod selected 
    '''
    xyz = [df['ycenter_target_centeriod'].values[0], df['zcenter_target_centeriod'].values[0]]
    pet_gt_prd_display = [ np.amax(pet, axis=0), np.amax(target_organ_aligned, axis=0), np.amax(gt, axis=0)]
    superimpose_segmentation_images(pet_gt_prd_display, file_name=str(case_name)+'_0', xyz=xyz)
    '''
    
    # dislay the axisl view of the spleen centeriod selected 
    '''
    xyz = [df['xcenter_target_centeriod'].values[0], df['ycenter_target_centeriod'].values[0]]
    pet_gt_prd_display = [ np.amax(pet, axis=2), 
                          np.amax(target_organ_aligned, axis=2),
                          np.amax(gt, axis=2)]
    superimpose_segmentation_images(pet_gt_prd_display, 
                                    file_name=str(case_name)+'_2',
                                      xyz=xyz)
    '''

    
    length = len(df)
    
    for row in range(length):
        calculated_features['ID'].append(str(case_name))
        calculated_features['x_psize'].append(res_selected[0])
        calculated_features['y_psize'].append(res_selected[1])
        calculated_features['z_psize'].append(res_selected[2])
        calculated_features['spleen_lesion_overlap'].append(spleen_lesion_overlap)
        
            
    df_update = pd.DataFrame(data=calculated_features)
    df = pd.concat([df, df_update], axis=1)        
    return df, nospleenseg


# function to read .nii and compute biomarker values
def read_nii_pet_and_mask_compute_features_save_csv(input_path, output_path, data_id ="data", allmaskinonefolder=False):
    """
    :param input_path: Path to the directory that consists the directory for .nii files
    :param output_path: The directory to save csv files, after computing the TMTV and Dmax, pixel spacing
    :return: read .nii, compute TMTV, Dmax, sDLS, LS, and other image-based biomarkers
    """
    case_ids = os.listdir(input_path)
    # case_ids = case_ids[:20]
    print("Total number of cases to read: 0.1%d", len(case_ids))

    calculated_features = defaultdict(list)
    calculated_bio = defaultdict(list)

    workers = []
    not_read = []
    df = pd.DataFrame()
    nospleensegall = defaultdict(list)
    for n, case_name in tqdm(enumerate(case_ids), total= len(case_ids)):
        # if case_name not in ['11009101056015','11009101056030','11009101116001','11009101436002','11009101436016' ,'11009101436019','11009102176001',
        #                      '11009104346006','61009225116011']:
        #     continue
        # try:
        df_update, nospleenseg = worker(input_path=input_path, case_name=case_name)
        for key, item in nospleenseg.items():
            nospleensegall[key].append(item)

        df = pd.concat([df, df_update])
        # except:
        #     not_read.append(case_name)

    # file saving as csv files 
    print("\n Cases could not read : total: ", len(not_read), " and are \n", not_read) 
    # print(df.head())
    # print(df.describe())
    name = os.path.join(output_path, str(data_id) + "_" +  str(time.time()) + '_biomarker.csv')
    df.to_csv(name, index=False, header=True)
    print(f"Extracted but not refined features saving at: {name} !!")
    return name, nospleensegall




if __name__ == "__main__":
    # List of functions implemented and testing stage 
    print(" Well come ")
    # worker()
    # get_features()
    # superimpose_segmentation_images()
    # read_nii_pet_and_mask_compute_features_save_csv()
