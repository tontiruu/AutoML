import lightgbm as lgb
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics


def model_lightgbm(analytic_type,target,df):
    #CはClassificationの略
    if analytic_type == "C":
        model = lgb.LGBMClassifier()
    else:
        model = lgb.LGBMRegressor()

   
    #文字データの判定
    for colum in (df.columns):
        if df[colum].dtype not in ["int64","float64"]:
        #ラベルエンコーディング
            from sklearn.preprocessing import LabelEncoder
            le=LabelEncoder()
            df[colum]=le.fit_transform(df[colum])

    print(df)
    y = df[target]
    x = df.drop([target], axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)

    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)


    # モデルの学習

    # パラメータ
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary', # 目的 : 分類
        'metric': {'auc'},
         'num_leaves': 20,
        'max_depth':5,
        'min_data_in_leaf': 3,
        'num_iteration': 100,
        'learning_rate':0.03,
        'num_boost_round':100,
         'early_stopping_rounds':20,
}
    model = lgb.train(
        params,
        train_set=lgb_train, # トレーニングデータの指定
        valid_sets=lgb_eval, # 検証データの指定
                  )
    
    #モデルの予測
    y_pred = model.predict(x_test)
    print(y_pred)

    accuracy = metrics.accuracy_score(y_test,np.where(y_pred > 0.5,1,0))

    #モデルの精度
    # accuracy = metrics.accuracy_score(y_test, y_pred) #予測値の精度を表す
    print(accuracy)


df=pd.read_csv("./templates/Data.csv")
print(df)

model_lightgbm("C","Target",df)


