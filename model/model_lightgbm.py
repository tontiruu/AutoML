import lightgbm as lgb

def model_lightgbm(analytic_type,target,df):
    #CはClassificationの略
    if analytic_type == "C":
        pass
    else:
        model = lgb.LGBMRegressor()


