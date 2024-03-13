import lightgbm as lgb
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn import metrics


class model_lightgbm:

    def learning(self,analytic_type,target,df):
        #CはClassificationの略
        self.analytic_type=analytic_type
        self.target=target
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
        self.model = lgb.train(
            params,
            train_set=lgb_train, # トレーニングデータの指定
            valid_sets=lgb_eval, # 検証データの指定
                    )
        
        
        #モデルの予測
        y_pred = self.model.predict(x_test)
        self.accuracy = metrics.accuracy_score(y_test,np.where(y_pred > 0.5,1,0))
        
        imp = list(self.model.feature_importance())
        imp = list(map(lambda x: round(x/sum(imp) * 100),imp))
        columns = list(x.columns)
        data = []
        for i,c in zip(imp,columns):
            data.append([i,c])
        data = sorted(data,reverse=True)

        self.imp = []
        self.columns = []
        for d in data:
            self.imp.append(d[0])
            self.columns.append(d[1])
            


        #モデルの精度
        # accuracy = metrics.accuracy_score(y_test, y_pred) #予測値の精度を表す
    
    def predict(self,pred_df):

        if self.taarget in pred_df.columns:
            pred_df = pred_df.drop([self.target], axis=1)#targetがあるなら落とす
        pred = self.model.predict(pred_df)#モデルを使って予測
        pred_df['target'] = pred#予測データをpred_dfに入れる

        return pred_df







