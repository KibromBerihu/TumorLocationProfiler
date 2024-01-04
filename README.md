# TumorLocationProfiler


### *TumorLocationProfiler: An AI-powered characterization of tumor locations relative to a reference organ. Paper [here](https://jnm.snmjournals.org/content/early/2023/12/07/jnumed.123.266322).*

  This project is **still under development**

***Introduction:***
We developed an AI-powered framework to characterize tumor location in relation to a reference organ on baseline whole-body PET/CT images.
<!-- Briefly, first, we used a publicly available AI model called ![TotalSegmentator](https://github.com/wasserth/TotalSegmentator), to automatically segment the spleen from baseline CT images.
A post-processing was developed to ensure the quality of the spleen segmentation.-->

Initially, we employed a publicly available AI model, [TotalSegmentator](https://github.com/wasserth/TotalSegmentator), to automatically segment the spleen from baseline CT images. 
Subsequently, a post-processing method was developed to ensure the quality of the spleen segmentation, as **quality control**.


Second, the tumor locations identified and delineated by experts on baseline PET/CT images were overlapped with the spleen segmentations. Following this, new biomarkers were calculated to characterize the tumor's location relative to the spleen. 
The new biomarkers are shown in **Figure 1**. Specifically, we systematically measured the distance between the centroid of the spleen and all other lesions, defining the standard deviation (SD) of these distances as
the lesion spread (**SpreadSpleen**). We calculated the maximum distance between the spleen and another lesion (**Dspleen**) for each patient.

![flow-digaram](https://github.com/KibromBerihu/TumorLocationProfiler/blob/main/images/graphical-abstract.png)

*Figure 1: graphical abstract of the proposed framework to characterize tumor location relative to a reference organ. [Kaplan-Meier survival analysis](https://lifelines.readthedocs.io/en/latest/Survival%20analysis%20with%20lifelines.html) of the overall and progression-free survival (OS and PFS respectively) are shown for both Dspleen and SpreadSpleen.*


In addition to the new biomarkers, other known biomarkers such as the dissemination (Dmax) (the distance between two farthest lesions), the distance between the largest lesion and another lesion (Dbulk), and total metabolic tumor volume (TMTV) were also calculated.
The age-adjusted international prognostic index (IPI) was provided. The predictive power of all features in terms of the progression-free and overall survival (PFS and OS, respectively) was calculated.

The added predictive values of the new biomarkers when they are integrated into a Cox model on the basis of TMTV, IPI, or TMTV combined with IPI were analyzed. Interestingly, their predictive power of the PFS and OS consistently and significantly improved.

**Good news**; It signifies that the new biomarkers provide complementary information to both IPI and TMTV.


please refer to the [paper](https://jnm.snmjournals.org/content/early/2023/12/07/jnumed.123.266322) for details and cite it if you use this for your research. 

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
      
               
To use the TotalSegmentator follow [jupyter notebook.](./totalsegmentator/TotalSegmentator.ipynb) and **TotalSegmentator** [documentation](https://github.com/wasserth/TotalSegmentator)

To extract the biomarkers follow [jupyter notebook.](./feature_extractor/spleen_based_features_extraction.ipynb)

## 📖 Citations 
Please cite the following papers if you use this package for your research:
```
Girum, KB, et al. "Tumor location relative to the spleen is a prognostic factor in lymphoma patients: a demonstration from the REMARC trial" The Journal of Nuclear Medicine (2023).

Wasserthal, Jakob, et al. "Totalsegmentator: Robust segmentation of 104 anatomic structures in ct images." Radiology: Artificial Intelligence 5.5 (2023).
```

## 🙏 Acknowledgments
We thank you [the reader].  
