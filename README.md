# TumorLocationProfiler

### under construction .....


### *TumorLocationProfiler: An AI-poweered characterization of tumor locations relative to a reference organ.*


***Introduction:***
We developed an AI-powered framework to characterize tumor location in relation to a reference organ on baseline whole-body PET/CT images.
<!-- Briefly, first, we used a publicly available AI model called ![TotalSegmentator](https://github.com/wasserth/TotalSegmentator), to automatically segment the spleen from baseline CT images.
A post-processing was developed to ensure the quality of the spleen segmentation.-->

Initially, we employed a publicly available AI model, ![TotalSegmentator](https://github.com/wasserth/TotalSegmentator), to automatically segment the spleen from baseline CT images. 
Subsequently, a post-processing method was developed to ensure the quality of the spleen segmentation, as **quality control**.


Second, the tumor locations delineated by experts on baseline PET/CT images were overlapped with the spleen segmentations. Following this, new biomarkers were calculated to characterize the tumor's location relative to the spleen.

In addition to the new biomarkers, we calculated other known biomarkers such as the dissemination (Dmax) (the distance between two farthest lesions), the distance between the largest lesion and another lesion (Dbulk), and total metabolic tumor volume (TMTV) were also calculated.
The age-adjusted international prognostic index (IPI) was provided. A comparison between all features in terms of their predictive power of the progression-free and overall survival (PFS and OS, respectively) was calculated.

The added predictive values of the new biomarkers when they are integrated into a Cox model on the basis of TMTV, IPI, or TMTV combined with IPI were analyzed. Interestingly, their predictive power of the PFS and OS consistently and significantly improved.

**Good news**; It signifies that the new biomarkers provide complementary information to both IPI and TMTV.



![flow-digaram](https://github.com/KibromBerihu/TumorLocationProfiler/blob/main/images/graphical-abstract.png)

*Figure 1: graphical abstract of the proposed framework to characterize tumor location relative to a reference organ.*

please refer to the paper for details and cite the paper if you use this for your research. 

### Table of contents  
- [Summary](#introduction)
- [Table of Contents](#table-of-contents)
- [ Required folder structure](#-required-folder-structure)
- [Installation](#installation)
- [Citations](#-citations)
- [Acknowledgements](#-acknowledgments)



## 📁 Required folder structure
Please provide all data in a single directory. The method automatically analyses all given data batch-wise. 

A typical data directory might look like:

    |-- main_folder                                             <-- The main folder or all patient folders (Give it any NAME)

    |      |-- parent folder (patient_folder_1)             <-- Individual patient folder name with unique id
    |           |-- CT                                     <-- The CT folder for the .nii file
                     | -- name.nii or name.nii.gz            <-- The CT image in nifti format (Name can be anything)
    |           |-- PET                                      <-- The corresponding pet folder for the .nii file  
                     | -- name.nii or name.nii.gz            <-- The pet image in nifti format (Name can be anything)
    |           |-- gt                                      <-- The corresponding ground truth folder for the .nii file  
                     | -- name.nii or name.nii.gz            <-- The ground truth (gt) image in nifti format (Name can be anything)
                     
    |      |-- parent folder (patient_folder_2)             <-- Individual patient folder name with unique id
    |          |-- CT                                     <-- The CT folder for the .nii file
                    | -- name.nii or name.nii.gz            <-- The cT image in nifti format (Name can be anything)
    |         |-- PET                                      <-- The corresponding pet folder for the .nii file  
                    | -- name.nii or name.nii.gz            <-- The pet image in nifti format (Name can be anything)
    |           |-- gt                                      <-- The corresponding ground truth folder for the .nii file  
                     | -- name.nii or name.nii.gz            <-- The ground truth (gt) image in nifti format (Name can be anything)                                         
    |           .
    |           .
    |           .
    |      |-- parent folder (patient_folder_N)             <-- Individual patient folder name with unique id
    |           |-- CT                                     <-- The CT folder for the .nii file
                    | -- name.nii or name.nii.gz            <-- The CT image in nifti format (Name can be anything)
    |           |-- PET                                      <-- The corresponding pet folder for the .nii file  
                    | -- name.nii or name.nii.gz            <-- The pet image in nifti format (Name can be anything)
    |           |-- gt                                      <-- The corresponding ground truth folder for the .nii file  
                     | -- name.nii or name.nii.gz            <-- The ground truth (gt) image in nifti format (Name can be anything)                    
                    
## ⚙️  Installation <a name="installation"> </a>
  

<font size='4'> Download/clone code to your local computer </font> 


    - git clone https://github.com/KibromBerihu/TumorLocationProfiler.git
   
    - Alternatively:
      1. go to https://github.com/KibromBerihu/TumorLocationProfiler.git >> [Code] >> Download ZIP file.
               
To use the TotalSegmentator follow [jupyter notebook.](./totalsegmentator/TotalSegmentator.ipynb) and [TotalSegmentator documentation](https://github.com/wasserth/TotalSegmentator)
To extract the biomarkers follow [jupyter notebook.](./features_extractor/spleen_based_features_extraction.ipynb)

## 📖 Citations 
Please cite the following papers if you use this package for your research:
```
Girum KB, Cottereau A-S, et al. Tumor location relative to the spleen is a prognostic factor in lymphoma patients: a demonstration from the REMARC trial
```

## 🙏 Acknowledgments
We thank you [the reader].  
