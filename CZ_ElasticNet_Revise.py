# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import scipy.io as sio
from sklearn import linear_model
from sklearn import preprocessing
from joblib import Parallel, delayed

def ElasticNet_3Fold_Permutation(Training_Data_Fold12, Training_Score_Fold12, Testing_Data_Fold3, Testing_Score_Fold3, \
                Training_Data_Fold13, Training_Score_Fold13, Testing_Data_Fold2, Testing_Score_Fold2, \
                Training_Data_Fold23, Training_Score_Fold23, Testing_Data_Fold1, Testing_Score_Fold1, \
                Times_IDRange, Alpha_Range, L1_ratio_Range, ResultantFolder, Parallel_Quantity, Max_Queued, QueueOptions):
    if not os.path.exists(ResultantFolder):
        os.mkdir(ResultantFolder)
    Subjects_Data_Mat = {'Training_Data_Fold12': Training_Data_Fold12, 'Testing_Data_Fold3': Testing_Data_Fold3, \
                         'Training_Data_Fold13': Training_Data_Fold13, 'Testing_Data_Fold2': Testing_Data_Fold2, \
                         'Training_Data_Fold23': Training_Data_Fold23, 'Testing_Data_Fold1': Testing_Data_Fold1}
    Subjects_Data_Mat_Path = ResultantFolder + '/Subjects_Data.mat'
    sio.savemat(Subjects_Data_Mat_Path, Subjects_Data_Mat)
    Subjects_Score_Mat = {'Training_Score_Fold12': Training_Score_Fold12, 'Testing_Score_Fold3': Testing_Score_Fold3, \
                          'Training_Score_Fold13': Training_Score_Fold13, 'Testing_Score_Fold2': Testing_Score_Fold2, \
                          'Training_Score_Fold23': Training_Score_Fold23, 'Testing_Score_Fold1': Testing_Score_Fold1}
    Subjects_Score_Mat_Path = ResultantFolder + '/Subjects_Score.mat'
    sio.savemat(Subjects_Score_Mat_Path, Subjects_Score_Mat)

    Finish_File = []
    Times_IDRange_Todo = np.int64(np.array([]))
    for i in np.arange(len(Times_IDRange)):
        ResultantFolder_I = ResultantFolder + '/Time_' + str(Times_IDRange[i])
        if not os.path.exists(ResultantFolder_I):
            os.mkdir(ResultantFolder_I)
        if not os.path.exists(ResultantFolder_I + '/Res_NFold.mat'):
            Times_IDRange_Todo = np.insert(Times_IDRange_Todo, len(Times_IDRange_Todo), Times_IDRange[i])
            Configuration_Mat = {'Subjects_Data_Mat_Path': Subjects_Data_Mat_Path, 'Subjects_Score_Mat_Path': Subjects_Score_Mat_Path, \
                'Alpha_Range': Alpha_Range, 'L1_ratio_Range': L1_ratio_Range, 'ResultantFolder_I': ResultantFolder_I, 'Parallel_Quantity': Parallel_Quantity};
            sio.savemat(ResultantFolder_I + '/Configuration.mat', Configuration_Mat)
            system_cmd = 'python3 -c ' + '\'import sys;\
                sys.path.append("/lustre/gaolab/cuizaixu/Utilities_Zaixu/Utilities_Regression/ElasticNet");\
                from CZ_ElasticNet_Revise import ElasticNet_3Fold_Permutation_Sub;\
                import os;\
                import scipy.io as sio;\
                configuration = sio.loadmat("' + ResultantFolder_I + '/Configuration.mat");\
                Subjects_Data_Mat_Path = configuration["Subjects_Data_Mat_Path"];\
                Subjects_Score_Mat_Path = configuration["Subjects_Score_Mat_Path"];\
                Alpha_Range = configuration["Alpha_Range"];\
                L1_ratio_Range = configuration["L1_ratio_Range"];\
                ResultantFolder_I = configuration["ResultantFolder_I"];\
                Parallel_Quantity = configuration["Parallel_Quantity"];\
                ElasticNet_3Fold_Permutation_Sub(Subjects_Data_Mat_Path[0], Subjects_Score_Mat_Path[0], Alpha_Range[0], L1_ratio_Range[0], ResultantFolder_I[0], Parallel_Quantity[0][0])\' ';
            system_cmd = system_cmd + ' > "' + ResultantFolder_I + '/perm_' + str(Times_IDRange[i]) + '.log" 2>&1\n'
            Finish_File.append(ResultantFolder_I + '/Res_NFold.mat')
            script = open(ResultantFolder_I + '/script.sh', 'w')  
            script.write(system_cmd)
            script.close()

    Jobs_Quantity = len(Finish_File)

    if len(Times_IDRange_Todo) > Max_Queued:
        Submit_Quantity = Max_Queued
    else:
        Submit_Quantity = len(Times_IDRange_Todo)
    for i in np.arange(Submit_Quantity):
        ResultantFolder_I = ResultantFolder + '/Time_' + str(Times_IDRange_Todo[i])
        Option = ' -V -o "' + ResultantFolder_I + '/perm_' + str(Times_IDRange_Todo[i]) + '.o" -e "' + ResultantFolder_I + '/perm_' + str(Times_IDRange_Todo[i]) + '.e"';
        os.system('qsub ' + ResultantFolder_I + '/script.sh ' + QueueOptions + ' -N perm_' + str(Times_IDRange_Todo[i]) + Option)
    if len(Times_IDRange_Todo) > Max_Queued:
        Finished_Quantity = 0;
        while 1:
            for i in np.arange(len(Finish_File)):
                if os.path.exists(Finish_File[i]):
                    Finished_Quantity = Finished_Quantity + 1
                    print(Finish_File[i])            
                    del(Finish_File[i])
                    print(time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())))
                    print('Finish quantity = ' + str(Finished_Quantity))
                    time.sleep(8)
                    ResultantFolder_I = ResultantFolder + '/Time_' + str(Times_IDRange_Todo[Submit_Quantity])
                    Option = ' -V -o "' + ResultantFolder_I + '/perm_' + str(Times_IDRange_Todo[Submit_Quantity]) + '.o" -e "' + ResultantFolder_I + '/perm_' + str(Times_IDRange_Todo[Submit_Quantity]) + '.e"';
                    cmd = 'qsub ' + ResultantFolder_I + '/script.sh ' + QueueOptions + ' -N perm_' + str(Times_IDRange_Todo[Submit_Quantity]) + Option
                    # print(cmd)
                    os.system(cmd)
                    Submit_Quantity = Submit_Quantity + 1
                    break
            if Submit_Quantity >= Jobs_Quantity:
                break

def ElasticNet_3Fold_Permutation_Sub(Subjects_Data_Mat_Path, Subjects_Score_Mat_Path, Alpha_Range, L1_ratio_Range, \
                ResultantFolder, Parallel_Quantity):
    # Data
    data = sio.loadmat(Subjects_Data_Mat_Path)
    Training_Data_Fold12 = data['Training_Data_Fold12']
    Testing_Data_Fold3 = data['Testing_Data_Fold3']
    Training_Data_Fold13 = data['Training_Data_Fold13']
    Testing_Data_Fold2 = data['Testing_Data_Fold2']
    Training_Data_Fold23 = data['Training_Data_Fold23']
    Testing_Data_Fold1 = data['Testing_Data_Fold1']
    # Score
    score = sio.loadmat(Subjects_Score_Mat_Path)
    Training_Score_Fold12 = score['Training_Score_Fold12'][0]
    Testing_Score_Fold3 = score['Testing_Score_Fold3'][0]
    Training_Score_Fold13 = score['Training_Score_Fold13'][0]
    Testing_Score_Fold2 = score['Testing_Score_Fold2'][0]
    Training_Score_Fold23 = score['Training_Score_Fold23'][0]
    Testing_Score_Fold1 = score['Testing_Score_Fold1'][0]
    ElasticNet_3Fold(Training_Data_Fold12, Training_Score_Fold12, Testing_Data_Fold3, Testing_Score_Fold3, \
                Training_Data_Fold13, Training_Score_Fold13, Testing_Data_Fold2, Testing_Score_Fold2, \
                Training_Data_Fold23, Training_Score_Fold23, Testing_Data_Fold1, Testing_Score_Fold1, \
                Alpha_Range, L1_ratio_Range, ResultantFolder, Parallel_Quantity, 1)

def ElasticNet_3Fold(Training_Data_Fold12, Training_Score_Fold12, Testing_Data_Fold3, Testing_Score_Fold3, \
                Training_Data_Fold13, Training_Score_Fold13, Testing_Data_Fold2, Testing_Score_Fold2, \
                Training_Data_Fold23, Training_Score_Fold23, Testing_Data_Fold1, Testing_Score_Fold1, \
                Alpha_Range, L1_ratio_Range, ResultantFolder, Parallel_Quantity, Permutation_Flag):

    if not os.path.exists(ResultantFolder):
        os.mkdir(ResultantFolder)
    ResultantFolder_Sub1 = ResultantFolder + '/Fold12_training_Fold3_testing';
    ResultantFolder_Sub2 = ResultantFolder + '/Fold13_training_Fold2_testing';
    ResultantFolder_Sub3 = ResultantFolder + '/Fold23_training_Fold1_testing';

    Predict_Corr_Fold12_training_Fold3_testing, Predict_MAE_Fold12_training_Fold3_testing = ElasticNet_APredictB(Training_Data_Fold12, \
        Training_Score_Fold12, Testing_Data_Fold3, Testing_Score_Fold3, Alpha_Range, L1_ratio_Range, 3, ResultantFolder_Sub1, Parallel_Quantity, Permutation_Flag)
    Predict_Corr_Fold13_training_Fold2_testing, Predict_MAE_Fold13_training_Fold2_testing = ElasticNet_APredictB(Training_Data_Fold13, \
        Training_Score_Fold13, Testing_Data_Fold2, Testing_Score_Fold2, Alpha_Range, L1_ratio_Range, 3, ResultantFolder_Sub2, Parallel_Quantity, Permutation_Flag)
    Predict_Corr_Fold23_training_Fold1_testing, Predict_MAE_Fold23_training_Fold1_testing = ElasticNet_APredictB(Training_Data_Fold23, \
        Training_Score_Fold23, Testing_Data_Fold1, Testing_Score_Fold1, Alpha_Range, L1_ratio_Range, 3, ResultantFolder_Sub3, Parallel_Quantity, Permutation_Flag)

    Mean_Corr = np.mean([Predict_Corr_Fold12_training_Fold3_testing, Predict_Corr_Fold13_training_Fold2_testing, Predict_Corr_Fold23_training_Fold1_testing])
    Mean_MAE = np.mean([Predict_MAE_Fold12_training_Fold3_testing, Predict_MAE_Fold13_training_Fold2_testing, Predict_MAE_Fold23_training_Fold1_testing])
    Predict_result = {'Mean_Corr':Mean_Corr, 'Mean_MAE':Mean_MAE}
    sio.savemat(ResultantFolder+'/Res_NFold.mat', Predict_result)
    return

def ElasticNet_APredictB_Permutation(Training_Data, Training_Score, Testing_Data, Testing_Score, Times_IDRange, \
                Alpha_Range, L1_ratio_Range, Nested_Fold_Quantity, ResultantFolder, Parallel_Quantity, Max_Queued, QueueOptions):
    if not os.path.exists(ResultantFolder):
        os.mkdir(ResultantFolder)
    Subjects_Data_Mat = {'Training_Data': Training_Data, 'Testing_Data': Testing_Data}
    Subjects_Data_Mat_Path = ResultantFolder + '/Subjects_Data.mat'
    sio.savemat(Subjects_Data_Mat_Path, Subjects_Data_Mat)
    Subjects_Score_Mat = {'Training_Score': Training_Score, 'Testing_Score': Testing_Score}
    Subjects_Score_Mat_Path = ResultantFolder + '/Subjects_Score.mat'
    sio.savemat(Subjects_Score_Mat_Path, Subjects_Score_Mat)

    Finish_File = []
    Times_IDRange_Todo = np.int64(np.array([]))
    for i in np.arange(len(Times_IDRange)):
        ResultantFolder_I = ResultantFolder + '/Time_' + str(Times_IDRange[i])
        if not os.path.exists(ResultantFolder_I):
            os.mkdir(ResultantFolder_I)
        if not os.path.exists(ResultantFolder_I + '/APredictB.mat'):
            Times_IDRange_Todo = np.insert(Times_IDRange_Todo, len(Times_IDRange_Todo), Times_IDRange[i])
            Configuration_Mat = {'Subjects_Data_Mat_Path': Subjects_Data_Mat_Path, 'Subjects_Score_Mat_Path': Subjects_Score_Mat_Path, \
                'Alpha_Range': Alpha_Range, 'L1_ratio_Range': L1_ratio_Range, 'Nested_Fold_Quantity': Nested_Fold_Quantity, 'ResultantFolder_I': ResultantFolder_I, 'Parallel_Quantity': Parallel_Quantity};
            sio.savemat(ResultantFolder_I + '/Configuration.mat', Configuration_Mat)
            system_cmd = 'python3 -c ' + '\'import sys;\
                sys.path.append("/lustre/gaolab/cuizaixu/Utilities_Zaixu/Utilities_Regression/ElasticNet");\
                from CZ_ElasticNet_Revise import ElasticNet_APredictB_Permutation_Sub;\
                import os;\
                import scipy.io as sio;\
                configuration = sio.loadmat("' + ResultantFolder_I + '/Configuration.mat");\
                Subjects_Data_Mat_Path = configuration["Subjects_Data_Mat_Path"];\
                Subjects_Score_Mat_Path = configuration["Subjects_Score_Mat_Path"];\
                Alpha_Range = configuration["Alpha_Range"];\
                L1_ratio_Range = configuration["L1_ratio_Range"];\
                Nested_Fold_Quantity = configuration["Nested_Fold_Quantity"];\
                ResultantFolder_I = configuration["ResultantFolder_I"];\
                Parallel_Quantity = configuration["Parallel_Quantity"];\
                ElasticNet_APredictB_Permutation_Sub(Subjects_Data_Mat_Path[0], Subjects_Score_Mat_Path[0], Alpha_Range[0], L1_ratio_Range[0], Nested_Fold_Quantity[0][0], ResultantFolder_I[0], Parallel_Quantity[0][0])\' ';
            system_cmd = system_cmd + ' > "' + ResultantFolder_I + '/perm_' + str(Times_IDRange[i]) + '.log" 2>&1\n'
            Finish_File.append(ResultantFolder_I + '/APredictB.mat')
            script = open(ResultantFolder_I + '/script.sh', 'w')  
            script.write(system_cmd)
            script.close()

    Jobs_Quantity = len(Finish_File)

    if len(Times_IDRange_Todo) > Max_Queued:
        Submit_Quantity = Max_Queued
    else:
        Submit_Quantity = len(Times_IDRange_Todo)
    for i in np.arange(Submit_Quantity):
        ResultantFolder_I = ResultantFolder + '/Time_' + str(Times_IDRange_Todo[i])
        Option = ' -V -o "' + ResultantFolder_I + '/perm_' + str(Times_IDRange_Todo[i]) + '.o" -e "' + ResultantFolder_I + '/perm_' + str(Times_IDRange_Todo[i]) + '.e"';
        os.system('qsub ' + ResultantFolder_I + '/script.sh ' + QueueOptions + ' -N perm_' + str(Times_IDRange_Todo[i]) + Option)
    if len(Times_IDRange_Todo) > Max_Queued:
        Finished_Quantity = 0;
        while 1:
            for i in np.arange(len(Finish_File)):
                if os.path.exists(Finish_File[i]):
                    Finished_Quantity = Finished_Quantity + 1
                    print(Finish_File[i])            
                    del(Finish_File[i])
                    print(time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())))
                    print('Finish quantity = ' + str(Finished_Quantity))
                    time.sleep(8)
                    ResultantFolder_I = ResultantFolder + '/Time_' + str(Times_IDRange_Todo[Submit_Quantity])
                    Option = ' -V -o "' + ResultantFolder_I + '/perm_' + str(Times_IDRange_Todo[Submit_Quantity]) + '.o" -e "' + ResultantFolder_I + '/perm_' + str(Times_IDRange_Todo[Submit_Quantity]) + '.e"';
                    cmd = 'qsub ' + ResultantFolder_I + '/script.sh ' + QueueOptions + ' -N perm_' + str(Times_IDRange_Todo[Submit_Quantity]) + Option
                    # print(cmd)
                    os.system(cmd)
                    Submit_Quantity = Submit_Quantity + 1
                    break
            if Submit_Quantity >= Jobs_Quantity:
                break

def ElasticNet_APredictB_Permutation_Sub(Subjects_Data_Mat_Path, Subjects_Score_Mat_Path, AlphaRange, L1_ratio_Range, Nested_Fold_Quantity, ResultantFolder, Parallel_Quantity):
    # Data
    data = sio.loadmat(Subjects_Data_Mat_Path)
    Training_Data = data['Training_Data']
    Testing_Data = data['Testing_Data']
    # Score
    score = sio.loadmat(Subjects_Score_Mat_Path)
    Training_Score = score['Training_Score'][0]
    Testing_Score = score['Testing_Score'][0]
    ElasticNet_APredictB(Training_Data, Training_Score, Testing_Data, Testing_Score, AlphaRange, L1_ratio_Range, Nested_Fold_Quantity, ResultantFolder, Parallel_Quantity, 1)
                            
def ElasticNet_APredictB(Training_Data, Training_Score, Testing_Data, Testing_Score, AlphaRange, L1_ratio_Range, Nested_Fold_Quantity, ResultantFolder, Parallel_Quantity, Permutation_Flag):
    
    if not os.path.exists(ResultantFolder):
        os.mkdir(ResultantFolder)

    if Permutation_Flag:
        # If do permutation, the training scores should be permuted, while the testing scores remain
        # Fold12
        Training_Index_Random = np.arange(len(Training_Score))
        np.random.shuffle(Training_Index_Random)
        Training_Score = Training_Score[Training_Index_Random]
        Random_Index = {'Training_Index_Random': Training_Index_Random}
        sio.savemat(ResultantFolder + '/Random_Index.mat', Random_Index);

    # Select optimal alpha & L1_ratio using inner fold cross validation
    Optimal_Alpha, Optimal_L1_ratio = ElasticNet_OptimalAlpha_KFold(Training_Data, Training_Score, Nested_Fold_Quantity, AlphaRange, L1_ratio_Range, ResultantFolder, Parallel_Quantity)

    Scale = preprocessing.MinMaxScaler()
    Training_Data = Scale.fit_transform(Training_Data)
    Testing_Data = Scale.transform(Testing_Data)  
    
    clf = linear_model.ElasticNet(alpha=Optimal_Alpha, l1_ratio=Optimal_L1_ratio)
    clf.fit(Training_Data, Training_Score)
    Predict_Score = clf.predict(Testing_Data)

    Predict_Corr = np.corrcoef(Predict_Score, Testing_Score)
    Predict_Corr = Predict_Corr[0,1]
    Predict_MAE = np.mean(np.abs(np.subtract(Predict_Score, Testing_Score)))
    Predict_result = {'Test_Score':Testing_Score, 'Predict_Score':Predict_Score, 'Weight':clf.coef_, 'Predict_Corr':Predict_Corr, 'Predict_MAE':Predict_MAE, 'alpha':Optimal_Alpha, 'l1_ratio':Optimal_L1_ratio}
    sio.savemat(ResultantFolder+'/APredictB.mat', Predict_result)
    return (Predict_Corr, Predict_MAE)

def ElasticNet_OptimalAlpha_KFold(Training_Data, Training_Score, Fold_Quantity, Alpha_Range, L1_ratio_Range, ResultantFolder, Parallel_Quantity):
    
    Subjects_Quantity = len(Training_Score)
    # Sort the subjects score
    Sorted_Index = np.argsort(Training_Score)
    Training_Data = Training_Data[Sorted_Index, :]
    Training_Score = Training_Score[Sorted_Index]
    
    Inner_EachFold_Size = np.int(np.fix(np.divide(Subjects_Quantity, Fold_Quantity)))
    MaxSize = Inner_EachFold_Size * Fold_Quantity
    EachFold_Max = np.ones(Fold_Quantity, np.int) * MaxSize
    tmp = np.arange(Fold_Quantity - 1, -1, -1)
    EachFold_Max = EachFold_Max - tmp
    Remain = np.mod(Subjects_Quantity, Fold_Quantity)
    for j in np.arange(Remain):
    	EachFold_Max[j] = EachFold_Max[j] + Fold_Quantity
    
    Parameter_Combination_Quantity = len(Alpha_Range) * len(L1_ratio_Range)
    Inner_Corr = np.zeros((Fold_Quantity, Parameter_Combination_Quantity))
    Inner_MAE_inv = np.zeros((Fold_Quantity, Parameter_Combination_Quantity))

    for k in np.arange(Fold_Quantity):
        
        Inner_Fold_K_Index = np.arange(k, EachFold_Max[k], Fold_Quantity)
        Inner_Fold_K_Data_test = Training_Data[Inner_Fold_K_Index, :]
        Inner_Fold_K_Score_test = Training_Score[Inner_Fold_K_Index]
        Inner_Fold_K_Data_train = np.delete(Training_Data, Inner_Fold_K_Index, axis=0)
        Inner_Fold_K_Score_train = np.delete(Training_Score, Inner_Fold_K_Index)
        
        Scale = preprocessing.MinMaxScaler()
        Inner_Fold_K_Data_train = Scale.fit_transform(Inner_Fold_K_Data_train)
        Inner_Fold_K_Data_test = Scale.transform(Inner_Fold_K_Data_test)    
        
        Parallel(n_jobs=Parallel_Quantity,backend="threading")(delayed(ElasticNet_SubAlpha)(Inner_Fold_K_Data_train, Inner_Fold_K_Score_train, Inner_Fold_K_Data_test, Inner_Fold_K_Score_test, Alpha_Range, L1_ratio_Range, l, ResultantFolder) for l in np.arange(Parameter_Combination_Quantity))        
        for l in np.arange(Parameter_Combination_Quantity):
            print(l)
            Fold_l_Mat_Path = ResultantFolder + '/Fold_' + str(l) + '.mat';
            Fold_l_Mat = sio.loadmat(Fold_l_Mat_Path)
            Inner_Corr[k, l] = Fold_l_Mat['Fold_Corr'][0][0]
            Inner_MAE_inv[k, l] = Fold_l_Mat['Fold_MAE_inv']
            os.remove(Fold_l_Mat_Path)
            
        Inner_Corr = np.nan_to_num(Inner_Corr)

    Inner_Corr_Mean = np.mean(Inner_Corr, axis=0)
    Inner_Corr_Mean_norm = (Inner_Corr_Mean - np.mean(Inner_Corr_Mean)) / np.std(Inner_Corr_Mean)
    Inner_MAE_inv_Mean = np.mean(Inner_MAE_inv, axis=0)
    Inner_MAE_inv_Mean_norm = (Inner_MAE_inv_Mean - np.mean(Inner_MAE_inv_Mean)) / np.std(Inner_MAE_inv_Mean)
    Inner_Evaluation = Inner_Corr_Mean_norm + Inner_MAE_inv_Mean_norm
    
    Inner_Evaluation_Mat = {'Inner_Corr':Inner_Corr, 'Inner_MAE_inv':Inner_MAE_inv, 'Inner_Evaluation':Inner_Evaluation}
    sio.savemat(ResultantFolder + '/Inner_Evaluation.mat', Inner_Evaluation_Mat)
    
    Optimal_Combination_Index = np.argmax(Inner_Evaluation) 
    
    Optimal_Alpha_Index = np.int64(np.ceil((Optimal_Combination_Index + 1) / len(L1_ratio_Range))) - 1
    Optimal_Alpha = Alpha_Range[Optimal_Alpha_Index]
    Optimal_L1_ratio_Index = np.mod(Optimal_Combination_Index, len(L1_ratio_Range))
    Optimal_L1_ratio = L1_ratio_Range[Optimal_L1_ratio_Index]
    return (Optimal_Alpha, Optimal_L1_ratio)

def ElasticNet_SubAlpha(Training_Data, Training_Score, Testing_Data, Testing_Score, Alpha_Range, L1_ratio_Range, Parameter_Combination_Index, ResultantFolder):
    # The range of Parameter_Combination_Index is: 0----(len(Alpha_Range)*len(L1_ratio_Range)-1))
    # Calculating the alpha index and l1_ratio index from the parameter_combination_index
    Alpha_Index = np.int64(np.ceil((Parameter_Combination_Index + 1) / len(L1_ratio_Range))) - 1
    L1_ratio_Index = np.mod(Parameter_Combination_Index, len(L1_ratio_Range))
    clf = linear_model.ElasticNet(l1_ratio=L1_ratio_Range[L1_ratio_Index], alpha=Alpha_Range[Alpha_Index])
    clf.fit(Training_Data, Training_Score)
    Predict_Score = clf.predict(Testing_Data)
    Fold_Corr = np.corrcoef(Predict_Score, Testing_Score)
    Fold_Corr = Fold_Corr[0,1]
    Fold_MAE_inv = np.divide(1, np.mean(np.abs(Predict_Score - Testing_Score)))
    Fold_result = {'Fold_Corr': Fold_Corr, 'Fold_MAE_inv':Fold_MAE_inv}
    ResultantFile = ResultantFolder + '/Fold_' + str(Parameter_Combination_Index) + '.mat'
    sio.savemat(ResultantFile, Fold_result)

def ElasticNet_Weight(Subjects_Data, Subjects_Score, Alpha_Range, L1_ratio_Range, Nested_Fold_Quantity, ResultantFolder, Parallel_Quantity):

    if not os.path.exists(ResultantFolder):
        os.mkdir(ResultantFolder)

    # Select optimal alpha using inner fold cross validation
    Optimal_Alpha, Optimal_L1_ratio = ElasticNet_OptimalAlpha_KFold(Subjects_Data, Subjects_Score, Nested_Fold_Quantity, Alpha_Range, L1_ratio_Range, ResultantFolder, Parallel_Quantity)
    
    Scale = preprocessing.MinMaxScaler()
    Subjects_Data = Scale.fit_transform(Subjects_Data)
    clf = linear_model.ElasticNet(alpha=Optimal_Alpha, l1_ratio=Optimal_L1_ratio)
    clf.fit(Subjects_Data, Subjects_Score)
    Weight_result = {'Weight':clf.coef_, 'alpha':Optimal_Alpha, 'l1_ratio':Optimal_L1_ratio}
    sio.savemat(ResultantFolder + '/Weight.mat', Weight_result)
    return;
