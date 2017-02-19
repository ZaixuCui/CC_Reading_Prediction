# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 20:39:09 2015

@author: Cui
"""

import scipy.io as sio
import numpy as np
import sys
sys.path.append('/lustre/gaolab/cuizaixu/Utilities_Zaixu/Utilities_Regression/ElasticNet')
import CZ_ElasticNet_Revise

ParentFolder = '/lustre/gaolab/cuizaixu/DATA_HCP_Reading_Revise/Prediction_S500_S900New_S2/PicVocab_AgeAdj_All'

data = sio.loadmat(ParentFolder + '/PicVocab_AgeAdj_S500_All.mat')
PicVocab_AgeAdj_S500_All = data['PicVocab_AgeAdj_S500_All']
PicVocab_AgeAdj_S500_All = np.transpose(PicVocab_AgeAdj_S500_All)
PicVocab_AgeAdj_S500_All = PicVocab_AgeAdj_S500_All[0]
data = sio.loadmat(ParentFolder + '/PicVocab_AgeAdj_S900New_All.mat')
PicVocab_AgeAdj_S900New_All = data['PicVocab_AgeAdj_S900New_All']
PicVocab_AgeAdj_S900New_All = np.transpose(PicVocab_AgeAdj_S900New_All)
PicVocab_AgeAdj_S900New_All = PicVocab_AgeAdj_S900New_All[0]

GMV_S500_All_training_mat = sio.loadmat(ParentFolder + '/GMV_S500_All_training.mat')
GMV_S500_All_training = GMV_S500_All_training_mat['GMV_S500_All_training']
GMV_S900New_All_testing_mat = sio.loadmat(ParentFolder + '/GMV_S900New_All_testing.mat')
GMV_S900New_All_testing = GMV_S900New_All_testing_mat['GMV_S900New_All_testing']

ResultantFolder = ParentFolder + '/results'
CZ_ElasticNet_Revise.ElasticNet_APredictB(GMV_S500_All_training, PicVocab_AgeAdj_S500_All, GMV_S900New_All_testing, \
    PicVocab_AgeAdj_S900New_All, np.exp(np.linspace(-6,5,20)), np.linspace(0.2, 1, 10), 3, ResultantFolder, 60, 0)
     
    
    
    
        
    







