# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 20:39:09 2015

@author: Cui
"""

import scipy.io as sio
import numpy as np
import os
import sys
sys.path.append('/lustre/gaolab/cuizaixu/Utilities_Zaixu/Utilities_Regression/ElasticNet')
import CZ_ElasticNet_Revise 

ParentFolder = '/lustre/gaolab/cuizaixu/DATA_HCP_Reading_Revise/Prediction_HCP_Model_S2/PicVocab_AgeAdj_All_3Fold'
Behavioral_Data = sio.loadmat(ParentFolder + '/PicVocab_AgeAdj_All_Index_Sorted.mat')

GMV_Fold12_training_All_PicVocab_mat = sio.loadmat(ParentFolder + '/GMV_Fold12_training_All_PicVocab.mat')
Training_Data_Fold12 = GMV_Fold12_training_All_PicVocab_mat['GMV_Fold12_training_All_PicVocab']
Training_Score_Fold12 = Behavioral_Data['Fold12_PicVocab_AgeAdj']
Training_Score_Fold12 = np.transpose(Training_Score_Fold12)
Training_Score_Fold12 = Training_Score_Fold12[0]
GMV_Fold12_training_Fold3_testing_All_PicVocab_mat = sio.loadmat(ParentFolder + '/GMV_Fold12_training_Fold3_testing_All_PicVocab.mat')
Testing_Data_Fold3 = GMV_Fold12_training_Fold3_testing_All_PicVocab_mat['GMV_Fold12_training_Fold3_testing_All_PicVocab']
Testing_Score_Fold3 = Behavioral_Data['Fold3_PicVocab_AgeAdj']
Testing_Score_Fold3 = np.transpose(Testing_Score_Fold3)
Testing_Score_Fold3 = Testing_Score_Fold3[0]

GMV_Fold13_training_All_PicVocab_mat = sio.loadmat(ParentFolder + '/GMV_Fold13_training_All_PicVocab.mat')
Training_Data_Fold13 = GMV_Fold13_training_All_PicVocab_mat['GMV_Fold13_training_All_PicVocab']
Training_Score_Fold13 = Behavioral_Data['Fold13_PicVocab_AgeAdj']
Training_Score_Fold13 = np.transpose(Training_Score_Fold13)
Training_Score_Fold13 = Training_Score_Fold13[0]
GMV_Fold13_training_Fold2_testing_All_PicVocab_mat = sio.loadmat(ParentFolder + '/GMV_Fold13_training_Fold2_testing_All_PicVocab.mat')
Testing_Data_Fold2 = GMV_Fold13_training_Fold2_testing_All_PicVocab_mat['GMV_Fold13_training_Fold2_testing_All_PicVocab']
Testing_Score_Fold2 = Behavioral_Data['Fold2_PicVocab_AgeAdj']
Testing_Score_Fold2 = np.transpose(Testing_Score_Fold2)
Testing_Score_Fold2 = Testing_Score_Fold2[0]

GMV_Fold23_training_All_PicVocab_mat = sio.loadmat(ParentFolder + '/GMV_Fold23_training_All_PicVocab.mat')
Training_Data_Fold23 = GMV_Fold23_training_All_PicVocab_mat['GMV_Fold23_training_All_PicVocab']
Training_Score_Fold23 = Behavioral_Data['Fold23_PicVocab_AgeAdj']
Training_Score_Fold23 = np.transpose(Training_Score_Fold23)
Training_Score_Fold23 = Training_Score_Fold23[0]
GMV_Fold23_training_Fold1_testing_All_PicVocab_mat = sio.loadmat(ParentFolder + '/GMV_Fold23_training_Fold1_testing_All_PicVocab.mat')
Testing_Data_Fold1 = GMV_Fold23_training_Fold1_testing_All_PicVocab_mat['GMV_Fold23_training_Fold1_testing_All_PicVocab']
Testing_Score_Fold1 = Behavioral_Data['Fold1_PicVocab_AgeAdj']
Testing_Score_Fold1 = np.transpose(Testing_Score_Fold1)
Testing_Score_Fold1 = Testing_Score_Fold1[0]
      
CZ_ElasticNet_Revise.ElasticNet_3Fold_Permutation(Training_Data_Fold12, Training_Score_Fold12, Testing_Data_Fold3, Testing_Score_Fold3, \
    Training_Data_Fold13, Training_Score_Fold13, Testing_Data_Fold2, Testing_Score_Fold2, \
    Training_Data_Fold23, Training_Score_Fold23, Testing_Data_Fold1, Testing_Score_Fold1, \
    np.arange(1000), np.exp(np.linspace(-6,5,20)), np.linspace(0.2, 1, 10), ParentFolder+'/Permutation2', 1, 250, '-q fat8 -l nodes=1:ppn=1')       
        
            
        
    
    
    
    



