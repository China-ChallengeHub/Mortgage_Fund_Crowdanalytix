import pandas as pd
import numpy as np
import xgboost as xgb
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from sklearn.metrics import f1_score

class xgb_model():
    def __init__(self,xg_train_s ,xg_val,xg_test,xg_train,target_train_s,target_val,target_train ,Unique_ID,ratio = 0.21 ,nrounds = 2500):
        param = {}
        param['booster'] = 'gbtree'
        param['objective'] = 'binary:logistic'
        param["eval_metric"] = 'auc'
        param['eta'] = 0.025
        param['max_depth'] = 6
        param['min_child_weight']=1
        param['subsample']= 0.8
        param['colsample_bytree']=0.8
        param['silent'] = 1
        self.param = param
        self.nrounds = nrounds
        self.ratio = ratio
        print('params' , self.param)
        #self._model_training_iter_(xg_train_s,xg_val,target_train_s,target_val)
        self._model_training_(xg_train,xg_test,target_train,Unique_ID)

    def _model_training_iter_(self,xg_train,xg_val,target_train,target_val):
        print('start the training')
        self.model = xgb.train(self.param, xg_train, self.nrounds)
        print('training complete' , self.model)
        pred_train = self.model.predict(xg_train)
        self.prob_cutoff = self._fin_threshohld(pred_train)
        self.pred_train =  np.where(pred_train >self.prob_cutoff , 1,0)
        pred_val = self.model.predict(xg_val)   ; self.pred_val = np.where(pred_val> self.prob_cutoff , 1, 0 )
        self.target_train = target_train.values ; self.target_val = target_val.values
        print('prediction length' , (self.pred_train) , (self.pred_val) ,len(self.pred_train) , len(self.pred_val))
        print('target length' , (self.target_train) , (self.target_val) , len(self.target_train) , len(self.target_val))
        self.score_train = f1_score(self.target_train, self.pred_train)
        self.score_val = f1_score(self.target_val, self.pred_val)
        print('model results' , self.score_train , self.score_val)
        self._feature_imp_(self.model)

    def _model_training_(self,xg_train,xg_test,target_train, Unique_ID):
        print('start the training')
        self.model = xgb.train(self.param, xg_train, self.nrounds)
        print('training complete' , self.model)
        pred_train = self.model.predict(xg_train) ;
        self.prob_cutoff = self._fin_threshohld(pred_train)
        self.pred_train =  np.where(pred_train >self.prob_cutoff ,1 ,0)
        preds = self.model.predict(xg_test)   ; self.preds = np.where(preds > self.prob_cutoff , 'NOT FUNDED' ,'FUNDED')
        self.target_train = target_train.values ; self.Unique_ID = Unique_ID.values
        print('prediction length' , (self.pred_train) , (self.preds) ,len(self.pred_train) , len(self.preds))
        print('target length' , (self.target_train) , len(self.target_train))
        self.score_train = f1_score(self.target_train, self.pred_train)
        print('f1_score' , self.score_train  )
        self._feature_imp_(self.model)
        submit = pd.DataFrame(self.preds,self.Unique_ID).reset_index() ; submit.columns = ['Unique_ID' , 'Result_Predicted']
        self.submit = submit
        print('Finally! the submission file' , submit.head(2))

    def _fin_threshohld(self, pred_train):
        pred_sorted = sorted(pred_train)
        break_point = int((1-self.ratio)*len(pred_sorted))
        print(pred_sorted[break_point], break_point)
        return pred_sorted[break_point]

    def _feature_imp_(self,model):
        feature_imp = self.model.get_score(importance_type='gain')
        feature_imp = list(feature_imp.items()) ; feature_imp = pd.DataFrame(feature_imp , columns = ['Feature' , 'Gains'])
        feature_imp = feature_imp.sort_values(by = ['Gains'] , ascending = False)
        feature_imp.to_csv("feature_imp.csv", index=False)
        print('feature_imp' , feature_imp)


    def f1_score_single(y_true, y_pred):
        y_true = set(y_true)
        y_pred = set(y_pred)
        cross_size = len(y_true & y_pred)
        if cross_size == 0: return 0.
        p = 1. * cross_size / len(y_pred)
        r = 1. * cross_size / len(y_true)
        return 2 * p * r / (p + r)

    def f1_score(y_true, y_pred):
        return np.mean([f1_score_single(x, y) for x, y in zip(y_true, y_pred)])
