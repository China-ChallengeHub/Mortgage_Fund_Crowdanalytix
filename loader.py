import pandas as pd
import numpy as np
import os
from os.path import join , exists
import xgboost as xgb

class data_read():

    def __init__(self, data_dir = "/Users/ankur/Documents/Competitions/Mortgage_CA/Data"):
        train = pd.read_csv(os.path.join(data_dir , 'train.csv') , na_values=' ')
        test = pd.read_csv(os.path.join(data_dir , 'test.csv') , na_values=' ')
        print('imported data' , train.head(2) , len(train) , len(test))
        self.column = ['Unique_ID','MORTGAGE_NUMBER','PROPERTY_VALUE','MORTGAGE_PAYMENT','GDS','LTV','TDS',
        'AMORTIZATION','MORTGAGE_AMOUNT','RATE','MORTGAGE_PURPOSE','PAYMENT_FREQUENCY','PROPERTY_TYPE',
        'TERM','FSA','AGE_RANGE','GENDER','INCOME','INCOME_TYPE','NAICS_CODE','CREDIT_SCORE','RESULT']
        self.strings = ['MORTGAGE_PURPOSE','PAYMENT_FREQUENCY','PROPERTY_TYPE' , 'FSA','AGE_RANGE',
        'GENDER','NAICS_CODE','RESULT']
        self.floats = ['MORTGAGE_NUMBER','PROPERTY_VALUE','MORTGAGE_PAYMENT','GDS','LTV','TDS','AMORTIZATION',
        'MORTGAGE_AMOUNT','RATE','TERM','INCOME','INCOME_TYPE','CREDIT_SCORE']
        train.columns = self.column ; test.columns = self.column
        train['id'] = 1 ; test['id'] = 0
        train = self._missing_value(train) ; test = self._missing_value(test)
        test_manual = test[(test['PAYMENT_FREQUENCY'] != 'Monthly')] ; test_manual = test_manual[['Unique_ID']]
        test_manual['Result_Predicted'] = 'FUNDED'
        print(len(test_manual),test_manual.head(2))
        test = test[(test['PAYMENT_FREQUENCY'] == 'Monthly')]
        df = train.append(test , ignore_index = True)
        self.train = train ; self.test = test ; self.df = df ; self.test_manual = test_manual

    def _missing_value(self, data):
        for key in data.keys():
            if key in self.strings:
                data[key].fillna('FUNDED', inplace = True)
            if key in self.floats:
                data[key].fillna(0, inplace = True)
        return data

class data_loader():

    def __init__(self,train,df):
        self.target = 'RESULT'
        self.Unique_ID = 'Unique_ID'
        self.strings = ['MORTGAGE_PURPOSE','PROPERTY_TYPE' , 'FSA','AGE_RANGE',
        'GENDER','NAICS_CODE','AMORTIZATION','RATE','TERM','INCOME_TYPE','FSA']
        self.floats = ['PROPERTY_VALUE','MORTGAGE_PAYMENT','GDS','TDS',
        'MORTGAGE_AMOUNT','INCOME','CREDIT_SCORE']
        self.to_embed = ['MORTGAGE_PURPOSE','PROPERTY_TYPE' ,'AGE_RANGE','GENDER','FSA',
        'NAICS_CODE','RESULT']
        self.xgb_features = ['PROPERTY_VALUE','GDS','LTV','TDS','FSA',
        'AMORTIZATION','MORTGAGE_AMOUNT','RATE','MORTGAGE_PURPOSE','PROPERTY_TYPE',
        'TERM','AGE_RANGE','GENDER','INCOME','INCOME_TYPE','NAICS_CODE','CREDIT_SCORE']
        self._init_emb_dict_([df])
        self._init_emb_insert([df])
        self._init_encode_vars([df])
        self._init_feature_encode_str_float([df])
        self._init_float_interactions([df])
        train_s , test, validation , train= self._init_sample([df])
        self._init_string_encode_dict_([train_s]) ; self._init_float_encode_dict_([train_s])
        self._init_enc_insert([train_s , test, validation , train])
        self.xgb_features += self.enc_feature_list
        self.xgb_features += self.interaction_feature_list
        self.xgb_features += self.encoded_string_feature_list
        self.xgb_features += self.encoded_float_feature_list
        self.xgb_features = np.unique(self.xgb_features)
        train_s.head(100).to_csv("sample.csv", index = False)
        print( self.xgb_features , len(self.xgb_features) ,train_s.shape , test.shape, validation.shape , train.shape)
        self._init_model_data_(train_s , test, validation, train)

    def _init_emb_dict_(self,dataframes):
        emb_dict = {}
        for df in dataframes:
            for key in df.keys():
                if key in self.to_embed:
                    emb_dict[key] = np.unique(df[key])
        self.emb_dict = emb_dict
        print('unique embed' , self.emb_dict)

    def _init_emb_insert(self,dataframes):
        for df in dataframes:
            for key in self.emb_dict.keys():
                if key in df.keys():
                    df[key] = df[key].replace(list(self.emb_dict.get(key)) , list(range(len(self.emb_dict.get(key)))))
                    print('done with embedding' , key , len(self.emb_dict.get(key)))
        return df

    def _init_feature_encode_str_float(self,dataframes):
        enc_feature_list = []
        for df in dataframes:
            for key1 in self.strings:
                for key2 in self.floats:
                    df[key1 + '_' + key2 + '_fenc'] = df.groupby([key1])[key2].transform(np.mean)
                    #df[key1 + '_' + key2 + '_fsum'] = df.groupby([key1])[key2].transform(np.sum)
                    enc_feature_list += [key1 + '_' + key2 + '_fenc']
        self.enc_feature_list = enc_feature_list

    def _init_float_interactions(self,dataframes):
        interaction_feature_list = []
        for df in dataframes:
            for i, key1 in enumerate(self.floats):
                if(i+1 < len(self.floats)):
                    for key2 in self.floats[i+1:]:
                        df[key1 + '_' + key2 + '_interaction'] = df[key1]/df[key2]
                        df[key1 + '_' + key2 + '_interaction'].fillna(0, inplace = True)
                        interaction_feature_list += [key1 + '_' + key2 + '_interaction']
        self.interaction_feature_list = interaction_feature_list

    def _init_sample(self,dataframes):
        for df in dataframes:
            train = df[(df['id'] == 1)] ; test = df[(df['id'] == 0)]
            train_s = train.sample(frac = 0.7) ; validation = train.loc[~train.index.isin(train_s.index)]
        return train_s , test, validation, train

    def _init_encode_vars(self,dataframes):
        for df in dataframes:
            for key in df.keys():
                if key in self.strings:
                    df[key+'_string_enc'] = df[key]
                if key in self.floats:
                    df[key+'_float_enc'] = pd.qcut(df[key], 10, labels=False)

    def _init_string_encode_dict_(self,dataframes):
        encoded_string_feature_list = []
        string_emb_dict = {}
        string_encode_dict = {}
        for df in dataframes:
            for key in df.keys():
                if key in self.strings:
                    string_emb_dict[key +'_string_enc'] = np.unique(df[key + '_string_enc'])
                    string_encode_dict[key +'_string_enc'] = df.groupby([key +'_string_enc'])[self.target].mean()
                    encoded_string_feature_list += [key +'_string_enc']
        self.string_emb_dict = string_emb_dict ; self.string_encode_dict = string_encode_dict
        self.encoded_string_feature_list = encoded_string_feature_list

    def _init_float_encode_dict_(self,dataframes):
        encoded_float_feature_list = []
        float_emb_dict = {}
        float_encode_dict = {}
        for df in dataframes:
            for key in df.keys():
                if key in self.floats:
                    float_emb_dict[key +'_float_enc'] = np.unique(df[key +'_float_enc'])
                    float_encode_dict[key +'_float_enc'] = df.groupby([key +'_float_enc'])[self.target].mean()
                    encoded_float_feature_list += [key +'_float_enc']
        self.float_emb_dict = float_emb_dict ; self.float_encode_dict = float_encode_dict
        self.encoded_float_feature_list = encoded_float_feature_list

    def _init_enc_insert(self,dataframes):
        for df in dataframes:
            for key in self.string_emb_dict.keys():
                if key in df.keys():
                    df[key] = df[key].replace(list(self.string_emb_dict.get(key)) , list(self.string_encode_dict.get(key)))
            for key in self.float_emb_dict.keys():
                if key in df.keys():
                    df[key] = df[key].replace(list(self.float_emb_dict.get(key)) , list(self.float_encode_dict.get(key)))

    def _init_model_data_(self,train_s , test, validation , train):
        self.target_train_s = train_s[self.target] ; self.target_val = validation[self.target]
        self.target_train = train[self.target] ; self.Unique_ID = test[self.Unique_ID]
        train_s = train_s[self.xgb_features] ; test = test[self.xgb_features]
        validation = validation[self.xgb_features] ; train = train[self.xgb_features]
        train_s = train_s.as_matrix() ; test = test.as_matrix()
        validation = validation.as_matrix() ; train = train.as_matrix()
        self.xg_train_s = xgb.DMatrix(train_s, label=self.target_train_s,feature_names=self.xgb_features)
        self.xg_val = xgb.DMatrix(validation, label=self.target_val,feature_names=self.xgb_features)
        self.xg_train = xgb.DMatrix(train, label=self.target_train,feature_names=self.xgb_features)
        self.xg_test = xgb.DMatrix(test,feature_names=self.xgb_features)
