from flask import Flask,render_template,redirect,request,url_for,session
import pandas as pd
from modules import createID
import time
from model.models import model_lightgbm
import math

app = Flask(__name__,static_folder="./static")



dataDict = {}
# key:id
# value:[timestamp,df]

accDict = {}
#key:id
#value:[timestamp,df]

allowedID = set([])

@app.route("/",methods=["POST","GET"])
def index():
    global allowedID
    id = createID.getID()
    while id in dataDict.keys():
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
        global dataDict
        print(request.files)
        file = request.files["CSVfile"]
        #'.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
        print(file)
        df = pd.read_csv(file)
        dataDict[id] = [time.time(),df]
        return redirect(f"/{id}/choseTarget")
    

@app.route("/<string:id>/choseTarget",methods = ["GET","POST"])
def choseTarget(id):
    if id not in allowedID:
        return redirect("/")
    df = dataDict[id][1]
    if request.method == "GET":
        columns = df.columns
        data = []
        for i in range(len(df)):
            data.append(list(df.iloc[i]))
        return render_template("choseTarget.html",columns=columns,data=data)
    else:
        checkedItem = request.form.getlist("checkbox")
        df = df[checkedItem]
        LGBM = model_lightgbm()
        LGBM.learning(analytic_type="C",target="Target",df = df)
        accuracy = LGBM.accuracy
        print(LGBM.imp)
        print(LGBM.columns)
        accDict[id] = [time.time(),accuracy]
        print(accuracy)
        return redirect(f"/{id}/score")


@app.route("/<string:id>/score",methods=["GET","POST"])
def score(id):
    if id not in allowedID:
        return redirect("/")
    if request.method == "GET":
        return render_template("score.html",accuracy = accDict[id][1], accpct = math.floor(accDict[id][1]*100))
    else:
        pass



if __name__ == '__main__':
    app.run()