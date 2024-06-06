from flask import Flask,render_template,redirect,request,url_for,session,make_response
import pandas as pd
from modules import createID
import time
from model.models import model_lightgbm
import math


app = Flask(__name__,static_folder="./static")


dataDictTrain = {}
# key:id
# value:[timestamp,df]

dataDictTest = {}
# key:id
# value:[timestamp,df]

modelDict = {}
#key:id
#value:[timestamp,imp,x_columns]

targetDict={}
#key:id
#value:target

analyticDict = {}

allowedID = set([])

@app.route("/",methods=["POST","GET"])
def index():
    global allowedID
    id = createID.getID()
    while id in dataDictTrain.keys():
        id = createID.getID()
    allowedID.add(id)
    if request.method == "GET":
        return render_template("index.html",id=id)


@app.route("/<string:id>/home",methods=["GET","POST"])
def home(id):
    if id not in allowedID:
        return redirect("/")
    if request.method == "GET":
        return render_template("home.html")
    else:
        global dataDictTrain, dataDictTest
        print(request.files)
        trainFile = request.files["trainingCSVfile"]
        testFile = request.files["testCSVfile"]
        #'.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
        
        try:
            dfTrain = pd.read_csv(trainFile)
            dataDictTrain[id] = [time.time(),dfTrain]
            dfTest = pd.read_csv(testFile)
            dataDictTest[id] = [time.time(),dfTest]
        except:
            return render_template("error.html", error_message="そのファイルは使用できません", return_page=f"/{id}/home")
        
        train_column=dfTrain.columns
        print(train_column)
        test_column=dfTest.columns
        print(test_column)

        



        return redirect(f"/{id}/chooseTarget")
    

@app.route("/<string:id>/chooseTarget",methods = ["GET","POST"])
def chooseTarget(id):
    if id not in allowedID:
        return redirect("/")
    dfTrain = dataDictTrain[id][1]
    dfTest = dataDictTest[id][1]
    if request.method == "GET":
        columns = dfTrain.columns
        data = []
        for i in range(min(30,len(dfTrain))):
            data.append(list(dfTrain.iloc[i]))
        return render_template("chooseTarget.html",columns=columns,data=data)
    else:
        #checkedItem = request.form.getlist("checkbox")
        #df = df[checkedItem]
        print(request.form)
        target = request.form["item"]
        analyticType = request.form["selection"]
        if analyticType == "分類":
            analyticType = "C"
            if len(set(list(dfTrain[target]))) > 100:
                return render_template("error.html",error_message="この目的変数は分類には適しません",return_page=f"/{id}/chooseTarget")
        else:
            analyticType = "L"

            if dfTrain[target].dtype not in ["int64","float64"]:
                return render_template("error.html", error_message="回帰分析では、目的変数を半角数字にしてください。", return_page=f"/{id}/chooseTarget")

        analyticDict[id]=analyticType
        targetDict[id]=target
        return redirect(f"/{id}/chooseColumns")


@app.route("/<string:id>/chooseColumns",methods=["GET","POST"])
def chooseColumns(id):
    if id not in allowedID:
        return redirect("/")
    dfTrain = dataDictTrain[id][1]
    dfTest = dataDictTest[id][1]
    if request.method == "GET":
        columns = list(dfTrain.columns)
        data = []
        for i in range(min(30,len(dfTrain))):
            data.append(list(dfTrain.iloc[i]))
        return render_template("chooseColumn.html",columns=columns,data=data,target=targetDict[id])
    else:
        #checkedItem = request.form.getlist("checkbox")
        #df = df[checkedItem]
        use_list=request.form.getlist("checkbox")
        use_set=set(use_list)
        use_set.add(targetDict[id])
        columns=list(dfTrain.columns)
        for column_name in columns:
            if column_name not in use_set:
                dfTrain=dfTrain.drop(column_name,axis=1)
                dfTest=dfTest.drop(column_name,axis=1)
        #モデルの学習。
        LGBM = model_lightgbm()
        LGBM.learning(analytic_type=analyticDict[id],target=targetDict[id],dfTrain = dfTrain,dfTest=dfTest)
        modelDict[id] = [time.time(),LGBM]
        return redirect(f"/{id}/score")

@app.route("/<string:id>/score",methods=["GET","POST"])
def score(id):
    if id not in allowedID:
        return redirect("/")
    if request.method == "GET":
        LGBM = modelDict[id][1]
        if analyticDict[id] == "C":
            score = LGBM.accuracy
        else:
            score = LGBM.MAE
        return render_template("score.html",score = score, accpct = math.floor(score*100),imp=LGBM.imp,columns = LGBM.columns,id =id, analyticType=analyticDict[id] ,y_test=LGBM.y_test,y_pred=LGBM.y_pred)
    else:
        pass



@app.route("/<string:id>/download",methods = ["GET","POST"])
def download(id):
    if id not in allowedID:
        return redirect("/")
    LGBM = modelDict[id][1]
    df = LGBM.pred_df
    csv = df.to_csv(encoding='utf-8-sig', index=False)

    response = make_response(csv)
    cd = 'attachment; filename=data.csv'
    response.headers['Content-Disposition'] = cd 
    response.mimetype='text/csv'
    return response

@app.route("/<string:id>/predict",methods = ["GET","POST"])
def predict(id):
    if id not in allowedID:
        return redirect("/")
    dfTest = dataDictTest[id][1]
    if request.method == "GET":
        LGBM = modelDict[id][1]
        LGBM.predict()

        pred_df = LGBM.pred_df
        columns = pred_df.columns
        data = []
        for i in range(min(30,len(pred_df))):
            data.append(list(pred_df.iloc[i]))
        return render_template("result.html",id = id,columns=columns,data=data)

@app.route("/terms_of_use",methods = ["GET","POST"])
def terms_of_use():
    if request.method == "GET":
        return render_template("terms_of_use.html")
    else:
        print('!')
        return redirect("/")

@app.route("/howtouse",methods=["GET"])
def howtouse():
    return render_template("howtouse.html")
if __name__ == '__main__':
    app.run()