# CC_Reading_Prediction

Functions and scripts for our CC paper. The functions contain elastic-net prediction (both 3F-CV and A-Predict-B) with inner 3F-CVs for parameter selection. Citing our paper will be greatly appreciated if you use these codes.
<br>&emsp; ```Zaixu Cui, Mengmeng Su, Liangjie Li, Hua Shu, Gaolang Gong; Individualized Prediction of Reading Comprehension Ability Using Gray Matter Volume, Cerebral Cortex, Volume 28, Issue 5, 1 May 2018, Pages 1656–1672, https://doi.org/10.1093/cercor/bhx061```

In this paper, scikit-learn (http://scikit-learn.org/stable/) 0.16.1 and python 3.4.3 were used to implement the elastic-net algorithm.

Also, the models construstruted using S500 dataset in our work was released to facilitate the full reproducibility of the paper.
The ORRT S500 model was in the folder 'ORRT_S500_All_Model' and the PVT S500 model was in the folder 'PVT_S500_All_Model'.
To use this ORRT model, you should have testing data matrix of m rows and 174947 columns.
If this variable is named testing_data, then:

from sklearn.externals import joblib

ss=joblib.load('Scale.pkl')

mm=joblib.load('Model.pkl')

testing_data_scaled = ss.transform(testing_data)

Prediction_Scores = mm.predict(testing_data_scaled)

To acquire the data with 174947 features, you should use our template for registration and our mask to extract the brain GMV voxels (http://gonglab.bnu.edu.cn/wp-content/resource/S500_All_DARTEL_Template_GMMask.rar).

## Note (important)

'Alpha' variable in the code is the 'lamda' in the second fomula of the paper.
'L1_ratio' variable in the code is the 'alpha' in the second fomula of the paper.

The resolution of GM mask and GMV were 2*2*2 in this study!
