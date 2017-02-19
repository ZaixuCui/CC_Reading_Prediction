# CC_Reading_Prediction

Functions and scripts for our CC paper (link will be added soon). The functions contain elastic-net prediction (both 3F-CV and A-Predict-B) with inner 3F-CVs for parameter selection.

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

To acquire the data with 174947 features, you should use our mask (http://gonglab.bnu.edu.cn/wp-content/resource/S500_All_DARTEL_Template_GMMask.rar) to extract the brain GMV voxels

