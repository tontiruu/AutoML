from flask import (
    Flask,
    render_template,
    redirect,
    request,
    url_for,
    session,
    make_response,
)
import pandas as pd
from modules import createID
import time
from model.models import model_lightgbm
import math


app = Flask(__name__, static_folder="./static")


dataDictTrain = {}
# key:id
# value:[timestamp,df]

dataDictTest = {}
# key:id
# value:[timestamp,df]

modelDict = {}
# key:id
# value:[timestamp,imp,x_columns]

allowedID = set([])


@app.route("/", methods=["POST", "GET"])
def index():
    global allowedID
    id = createID.getID()
    while id in dataDictTrain.keys():
        id = createID.getID()
    allowedID.add(id)
    if request.method == "GET":
        return render_template("index.html", id=id)


@app.route("/<string:id>/home", methods=["GET", "POST"])
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

        dfTrain = pd.read_csv(trainFile)
        dataDictTrain[id] = [time.time(), dfTrain]
        dfTest = pd.read_csv(testFile)
        dataDictTest[id] = [time.time(), dfTest]
        return redirect(f"/{id}/choseTarget")


@app.route("/<string:id>/choseTarget", methods=["GET", "POST"])
def choseTarget(id):
    if id not in allowedID:
        return redirect("/")
    dfTrain = dataDictTrain[id][1]
    dfTest = dataDictTest[id][1]
    if request.method == "GET":
        columns = dfTrain.columns
        data = []
        for i in range(len(dfTrain)):
            data.append(list(dfTrain.iloc[i]))
        return render_template("choseTarget.html", columns=columns, data=data)
    else:
        # checkedItem = request.form.getlist("checkbox")
        # df = df[checkedItem]
        print(request.form)
        target = request.form["item"]
        analyticType = request.form["selection"]
        if analyticType == "分類":
            analyticType = "C"
        else:
            analyticType = "L"

        LGBM = model_lightgbm()
        LGBM.learning(analytic_type="C", target=target, dfTrain=dfTrain, dfTest=dfTest)
        modelDict[id] = [time.time(), LGBM]
        return redirect(f"/{id}/score")


@app.route("/<string:id>/score", methods=["GET", "POST"])
def score(id):
    if id not in allowedID:
        return redirect("/")
    if request.method == "GET":
        LGBM = modelDict[id][1]
        return render_template(
            "score.html",
            accuracy=LGBM.accuracy,
            accpct=math.floor(LGBM.accuracy * 100),
            imp=LGBM.imp,
            columns=LGBM.columns,
        )
    else:
        pass


@app.route("/<string:id>/download", methods=["GET", "POST"])
def download(id):
    if id not in allowedID:
        return redirect("/")
    LGBM = modelDict[id][1]
    df = LGBM.pred_df
    csv = df.to_csv(index=False)

    response = make_response(csv)
    cd = "attachment; filename=data.csv"
    response.headers["Content-Disposition"] = cd
    response.mimetype = "text/csv"
    return response


@app.route("/<string:id>/predict", methods=["GET", "POST"])
def predict(id):
    if id not in allowedID:
        return redirect("/")
    dfTest = dataDictTest[id][1]
    if request.method == "GET":
        LGBM = modelDict[id][1]
        LGBM.predict()
        return render_template("result.html", id=id)


if __name__ == "__main__":
    app.run()
