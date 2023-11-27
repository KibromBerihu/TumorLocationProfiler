# load the important labraries. 
import os
from pathlib import Path
import numpy as np
import pandas as pd
import math 

from scipy.spatial import distance
from collections import defaultdict
import time 


# get features from the centiod of the tumor region and the reference organ
def calculate_euclidean_based_distance(xyz, xyz_ref, xyz_bulk):
    # calculate the euclidean distance between the given reference organ's centriod, xyz_ref, and center of tumor regions xyz, or
    # the distance between the largest lesion (bulk lesion) and other lesions, per patient
    # dissemination_all = [distance.euclidean(a, b) for a, b in zip(xyz, xyz_ref)]    
    
    # calculate the distance between two farthest lesions (dmax)                 
    dist_dmax = []  
    for k in range(len(xyz)):
        if len(xyz) == 1: # only one lesion detected 
            dist_dmax.append(0)
        else:
            a = xyz[k]
            # distance between the selected lesion's centeriod a and all other lesion centers 
            for kk in range(len(xyz)):
                if kk >k:
                    b = xyz[kk]
                    dmax_now = distance.euclidean(a, b)
                    dist_dmax.append(dmax_now)
  
    features = defaultdict(list)
    if not len(dist_dmax):
       dist_dmax = 0 

    features['Dmax_new'] = np.max(dist_dmax)
    features['Dstd'] = np.std(dist_dmax)

    # Calculate the distance between the largest lesion (bulk lesion) and the farhest lesion from the bulk
    dist_dbulk = []
    # Calculate the distance between a reference target (e.g., the spleen or the liver) and the farthest lesion from the target, and the spread of the lesion from the target
    dist_ref  = [] # distance of the lesions from the reference organ
    
    # bulk distance
    a = xyz_bulk[0]
    # reference target organ distance 
    ref_target = xyz_ref.copy()  

    for k in range(len(xyz)):
        if len(xyz) == 1:           
            dist_ref = distance.euclidean(xyz[k], ref_target[0])
            dist_dbulk.append(0) # one lesion 
        else:
            b = xyz[k]
            dbulk_now = distance.euclidean(a, b)
            dist_dbulk.append(dbulk_now)
            
            dref = distance.euclidean(b, ref_target[0])
            dist_ref.append(dref)
            
    features['Dbulk'] = np.max(dist_dbulk)
    features['Dbulk_std'] = np.std(dist_dbulk)

    # distance between the target organ and the farthest lesion
    features['DLS'] = np.max(dist_ref) 
    # get the coordinates of the distant lesion from the reference organ 
    k = np.array(dist_ref).argmax() 
    features['x_dspleen'] = xyz[k][0]
    features['y_dspleen'] = xyz[k][1]
    features['z_dspleen'] = xyz[k][2]

    # lesion spread: the standard deviation of all distances measured between the lesions and the target organ
    features['LS'] = np.std(dist_ref)   
     
    return features


def calcualte_features(df, data_id="dataX", output_path=None ):
    """
    Function to compute the image-based biomarkers 
    """          
    if not len(df):
        return 
    
    df_features = pd.DataFrame()


    # Spleen segmentation santiy check 
    # ref_xyz is the refernce organ, for example the spleen centeriod 
    error_detected_target_organ = []
    for patient_id in np.unique(df['ID'].values):
        df_selected = df.loc[df['ID'] == patient_id]
        ref_xyz = np.unique(df_selected[['xcenter_target', 'ycenter_target', 'zcenter_target']])
        ref_xyz = ref_xyz[~np.isnan(ref_xyz)]
        if np.sum(ref_xyz) == 0 or np.isnan(ref_xyz).any() or len(ref_xyz) !=3:
            error_detected_target_organ.append(patient_id)

    df1 = df.loc[~df["ID"].isin(error_detected_target_organ)]
    ref_xyz_voxel = df1[['xcenter_target', 'ycenter_target', 'zcenter_target', 'x_psize', 'y_psize', 'z_psize']].dropna().drop_duplicates()
    ref_xyz = df1[['xcenter_target', 'ycenter_target', 'zcenter_target' ]].dropna().drop_duplicates()

    xyz_median = ref_xyz.median().values
    
    ''' 
    print("reference target coordinates", ref_xyz) 
    print(f"medain organ center {xyz_median}, from {len(ref_xyz)} centers")
    print("target organ statistics, x, y, x respectively")
    print(ref_xyz_voxel['xcenter_target'].multiply(ref_xyz_voxel['x_psize']*0.1, axis='index').describe())
    print(ref_xyz_voxel['ycenter_target'].multiply(ref_xyz_voxel['y_psize']*0.1, axis='index').describe())
    print(ref_xyz_voxel['zcenter_target'].multiply(ref_xyz_voxel['z_psize']*0.1, axis='index').describe())
    print(f"Total cases {len(np.unique(df['ID']))}")
    '''

    # Compute the Spleen-based and lesion-based features 
    for patient_id in np.unique(df['ID'].values):
        df_selected = df.loc[df['ID'] == patient_id]
        xyz = (df_selected[['xcenter', 'ycenter', 'zcenter' ]].dropna()).values
        voxel = (df_selected[['x_psize', 'y_psize', 'z_psize' ]].dropna()).values
        xyz = np.multiply(xyz, voxel[:xyz.shape[0], :])*0.1
        ref_xyz = np.unique(df_selected[['xcenter_target',
                                        'ycenter_target', 'zcenter_target']])
        ref_xyz = ref_xyz[~np.isnan(ref_xyz)]

        ref_xyz_save = ref_xyz.copy()
        
        # check if the spleen was detected, if no center of lesions were detected, spleen not detected. 
        spleendetected = True 
        if np.sum(ref_xyz) == 0:
            print(f" {patient_id} no spleen detected ")
            ref_xyz = xyz_median
            spleendetected = False 

        elif np.isnan(ref_xyz).any():
            print(f" {patient_id} detected spleen has nan centeriod ")
            ref_xyz = xyz_median
            spleendetected = False 

            
            
        elif len(ref_xyz) !=3:
            print(f" {patient_id} more than one region were detected as spleen ")
            ref_xyz = xyz_median
            spleendetected = False 

            
        # multiplay the reference coordinates with the voxel space, it will be in cm 
        ref_xyz = np.array(np.multiply(ref_xyz, 
                                    voxel[0]))*0.1
        # add noise to the centeriod of the spleen
        '''
        # print(f"Reference target centeriod in cm: ", ref_xyz)

        # experiment to disrupt the centeriod of the spleen 
        iselect = np.random.randint(8)
        candidate = [[0, 0, 2], [0, 2, 0], [2, 0, 0], [-2, 0, 0], [0, -2, 0], [0, 0, -2], [1, 1, 0], [-1, -1, 0]]
        ref_xyz_modified = candidate[iselect] + ref_xyz
        print("modified by:",candidate[iselect])
        print(f"reference target centeriod modified", ref_xyz_modified)
        ref_xyz = ref_xyz_modified
        '''


        ref_xyz = [ref_xyz]

        # select the lesion with maxium tumor region and get its centeriod. 
        xyz_bulk = df_selected[df_selected['area'] == df_selected['area'].max()]
        voxel_space = df_selected[['x_psize', 'y_psize', 'z_psize' ]].values[:len(xyz_bulk[['xcenter', 'ycenter', 'zcenter' ]]), :]
        xyz_bulk = np.multiply(xyz_bulk[['xcenter', 'ycenter', 'zcenter' ]].values, voxel_space
                            )*0.1
        
        # Calculate the features based on the give centeriod of the lesions in xyz and reference target xyz_ref, and the bulk lesion centeriod
        # xyz_bulk in cm :
        if not len(xyz):
            print(df_selected['spleen_lesion_overlap'].values[0])
            print(patient_id)
        if len(xyz):
            features = calculate_euclidean_based_distance(xyz=xyz, xyz_ref=ref_xyz, xyz_bulk=xyz_bulk)
        else:
            features['Dbulk'] = 0
            features['Dbulk_std'] = 0

            features['DLS'] = 0
            features['LS'] = 0 


            features['Dmax_new'] = 0
            features['Dstd'] = 0

            features['x_dspleen'] = 0
            features['y_dspleen'] = 0
            features['z_dspleen'] = 0

        features['rx_dspleen'] = (features['x_dspleen']/ voxel[0][0])*10
        features['ry_dspleen'] = (features['y_dspleen']/ voxel[0][1])*10
        features['rz_dspleen'] = (features['z_dspleen']/ voxel[0][2])*10

        # print(ref_xyz_save)
        if len(ref_xyz_save):
            features['xcenter_target'] = ref_xyz_save [0]
            features['ycenter_target'] = ref_xyz_save [1]
            features['zcenter_target'] = ref_xyz_save [2]
        else:
            features['xcenter_target'] = math.nan
            features['ycenter_target'] = math.nan
            features['zcenter_target'] = math.nan
            print("Nan", patient_id)


        
        features['x_psize'] = df_selected[['x_psize']].values[0]
        features['y_psize'] = df_selected[['y_psize']].values[0]
        features['z_psize'] = df_selected[['z_psize']].values[0]
        
        features["ID"] = str(patient_id)
        features['TMTV'] = df_selected["TMTV"].values[0]*0.001
        features['SpleenDetected'] = spleendetected
        features['Ntumors'] = np.unique(df_selected['Ntumors'])
        features['spleen_lesion_overlap'] = df_selected['spleen_lesion_overlap'].values[0]
        if spleendetected:
            voxel_space = df_selected[['x_psize', 'y_psize', 'z_psize' ]].values[:3]
            voxel_space = voxel_space[0]
            features['spleen_area'] =  df_selected['center_area'].values[0]*voxel_space[0]*voxel_space[1]*voxel_space[2]*0.001
        else:
            features['spleen_area'] = float(math.nan)
 
        df_update = pd.DataFrame(data=features, index=[0])
        df_features = pd.concat([df_features, df_update], axis=0)


        
        
        
    df_features = df_features.set_index('ID')
    # print(df_features.describe())

    # save csv file 
    if output_path is None:
        # default path 
        output_path = r"../" 
    name = os.path.join(output_path, str(data_id) + "_" +  str(time.time()) + '_biomarker.csv')
    df_features.to_csv(name, index=True, header=True)
    print('Total number of patients correctly read and their volume calculated: ', len(df_features))
    print(f"Calculated featured saved to the file directory: {name} !!")
    
    return df_features 