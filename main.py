import pandas as pd
import numpy as np
import os
from loader import data_read , data_loader
from models import xgb_model

if __name__ == '__main__':
    print('yo! here we go')
    data = data_read()
    print('data columns' , data.df.columns)
    print('data head' , data.df.head(2))
    loader = data_loader(data.train , data.df)
    model = xgb_model(loader.xg_train_s ,loader.xg_val, loader.xg_test,loader.xg_train,
    loader.target_train_s,loader.target_val, loader.target_train , loader.Unique_ID)
    submit = model.submit.append(data.test_manual , ignore_index = True)
    submit.to_csv("submit.csv", index=False)
