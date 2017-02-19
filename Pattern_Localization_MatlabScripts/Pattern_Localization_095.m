
clear 

load Weight_New_095.mat;
Mask_Path = '/lustre/gaolab/cuizaixu/DATA_HCP_Reading_Revise/T1_S500_SegmentationDARTEL/S500_All/GMV_GMD_2mm_S2/All_S500_GM_Mask.nii';
% Resultant_File = '/lustre/gaolab/cuizaixu/DATA_HCP_Reading_Revise/Prediction_Weight_S2/PicVocab_AgeAdj_All/Weight.nii';
% Vector_to_Volume(Weight_New, Mask_Path, Resultant_File);
Resultant_File = '/lustre/gaolab/cuizaixu/DATA_HCP_Reading_Revise/Prediction_Weight_S2/PicVocab_AgeAdj_All/Weight_abs_095.nii';
Vector_to_Volume(abs(Weight_New), Mask_Path, Resultant_File);