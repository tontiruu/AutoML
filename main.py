#flaskというライブ来のインポート文。flaskはサーバーを立ち上げて、そのサーバーへの通信が来た時に、どんな処理をして、どのHTMLを表示するのかなどを決めることができる!
from flask import Flask,render_template,redirect,request,url_for,session,make_response

#csvからデータを読み込んで加工するためのライブラリ
import pandas as pd

#IDを付与するもの。modulesフォルダの中のcreateIDを参照してる。このIDでユーザーを区別することで、アップロードされたデータが混同しないようにしてる。
from modules import createID

#時間関係の操作ができるライブラリ。今回は現在時刻をタイムスタンプとして記録するために使う
import time

#modelフォルダーのmodelsに書かれてるmodel_lightgbmクラスをインポートしてmainでも使えるようにしてる。model_lightgbmは前処理とかも含めて一連の機械学習の流れを書いてある。
#main.pyは基本的に通信に対してどのデータをHTML側に渡して、何を表示するかなどを書いているだけで、機械学習自体はこのmodel_lightgbmクラスの中に書かれてる！
from model.models import model_lightgbm

#切り捨ての計算に使う
import math

#Flaskのインスタンス化　簡単に言うとFlaskの機能をappという変数に詰め込んでいるイメージ！
#static_folderにはCSSが入ってるファイルのパスを渡している。HTMLはflaskでは必ずHTMLファイルはtemplatesという名前のフォルダに入れるという決まりごとがあるから、とくに指定しなくても読み込んでくれる。
app = Flask(__name__,static_folder="./static")

#トレインデータをIDと紐づけて保存するための辞書型。辞書型は jisyo = {"a":130,"b":120}とすると、jisyo["a"]が130、jisyo["b"]で120になるようにデータの紐づけができる。また、古いデータは自動で削除できるようにデータアップロー時のタイムスタンプも記録する
dataDictTrain = {}
# key:id
# value:[timestamp,df]

#テストデータをIDで紐づけて保存するための辞書型
dataDictTest = {}
# key:id
# value:[timestamp,df]

#機械学習の結果等をIDで紐づけて保存するための辞書型
modelDict = {}
#key:id
#value:[timestamp,imp,x_columns]

#選んだ目的変数のカラム名をIDで紐づけて保存するための辞書型
targetDict={}
#key:id
#value:target

#回帰分析をするのか、分類をするのかをIDで紐づけて保存するための辞書型。
analyticDict = {}

#すでに発行したIDを記録しておくためのset型。set型は基本list型で、要素の重複は許さないようなイメージ！
allowedID = set([])

#https://sophiadatatech.pythonanywhere.com/にアクセスされたときの処理
@app.route("/",methods=["POST","GET"])
def index():
    global allowedID
    #データをユーザーごとに区別して保存できるようにするためのidの発行
    id = createID.getID()
    #万が一すでに発行されていたIDなら、発行されてないものになるまで再発行し続ける処理
    while id in allowedID:
        id = createID.getID()
    #発行済みのIDを記録しているset型変数に今発行したIDを追加する。
    allowedID.add(id)
    #もし通信のリクエストがGETだった時の処理（基本的に何かのURLにアクセスして、ページが表示されるような通信はGET）
    if request.method == "GET":
        #index.htmlを表示する処理。index.htmlないでも変数idを使えるように、id=idとして渡している。
        return render_template("index.html",id=id)


#https://sophiadatatech.pythonanywhere.com/ユーザーごとに割り振ったID/homeにアクセスされたときの処理
@app.route("/<string:id>/home",methods=["GET","POST"])
def home(id):
    #もし、発行済みじゃないIDが使われていた場合は最初のページに戻す（この処理がないとユーザーがいきなりhomeや、もっと先のページに適当なIDでアクセスできてしまい、バグの原因になる）
    if id not in allowedID:
        #https://sophiadatatech.pythonanywhere.com/に飛ばす処理
        return redirect("/")
    #もしリクエストがGETだった時の処理
    if request.method == "GET":
        #home.htmlを表示させる処理
        return render_template("home.html")
    #もし、リクエストがGETではない、つまりPOSTであった時の処理。POSTはユーザーが何かしらのフォームに入力を行い、サーバー側にデータを送信するときに使われる。今回のケースだと、ユーザーはファイルを入力し、サーバーに送信している。
    else:
        #基本,関数の中（今回でいうと、def home(id)のなか）では、関数の外で定義された変数を書き換えることができないが、globalを付けることで関数名での書き換えが可能になる。
        global dataDictTrain, dataDictTest
        #送られてきたデータのなかから、trainingCSVfileという名前の要素を取得し、trainFileに代入する。ここで指定している名前はhtmlで、<input name="名前">としたときの名前を入れる。
        trainFile = request.files["trainingCSVfile"]
        #送られてきたデータの中から、testCSVfileという名前のデータを取得し、testFIleに代入する。
        testFile = request.files["testCSVfile"]

        #try,except構文は、まずtryのなかの処理を実行し、正常に動いたらそのままexceptのなかは無視され次の処理に進む。もしエラーが起きた場合は、エラーでプログラムを終了させることなくexceptに書かれた処理を実行してから次へと進む。
        #今回は、そもそもCSVファイルでないものや、エンコーディングの問題によってデータを読み込めないことがあるので、try,exceptを使い、読み込めない時はエラーメッセージを表示させるようにしてる。
        try:
            #trainFileのデータを読み込み
            dfTrain = pd.read_csv(trainFile)
            #タイムスタンプと作成したデータフレイムをidに紐づけて保存。タイムスタンプを入れるのは古くなったデータを消したくて、その時に作成日時のデータが必要になりそうだから
            dataDictTrain[id] = [time.time(),dfTrain]
            #dfTestのデータの読み込み
            dfTest = pd.read_csv(testFile)
            #idにタイムスタンプとデータを紐づけて保存
            dataDictTest[id] = [time.time(),dfTest]
        except:
            #error.htmlを表示させる。この時にerror_messageとreturn_page（戻るボタンで飛ばされるリンク）の情報を渡す
            return render_template("error.html", error_message="そのファイルは使用できません", return_page=f"/{id}/home")
        #chooseTargetののページに飛ばす
        return redirect(f"/{id}/chooseTarget")
    
#https://sophiadatatech.pythonanywhere.com/ユーザーごとに割り振ったID/chooseTargetにアクセスされたときの処理
@app.route("/<string:id>/chooseTarget",methods = ["GET","POST"])
def chooseTarget(id):
    #すでに発行したIDではない場合ホームページに戻す
    if id not in allowedID:
        return redirect("/")
    #idとタイムスタンプ、データを紐づけた辞書型から、idをキーとしてデータを読み込む　dataDictTrain[id]で、[作成時のタイムスタンプ,データ]にアクセスできるから、さらに[1]を指定して2番目のデータをとってくる。(pythonでは[0]で一番目。[1]で二番目の要素にアクセスできる)
    dfTrain = dataDictTrain[id][1]
    #リクエストがGETの時の処理
    if request.method == "GET":
        #学習に使うデータのカラム名を取得
        columns = dfTrain.columns
        #html側に渡すデータを入れるdata変数を作成
        data = []
        #最大30回繰り返す。min(30,len(dfTrain))は、30とdfTrainのデータ数のうち、小さいほうを採用するという意味。最大30件に絞ってHTML側に渡す理由は、すべてのデータを渡すとデータ数が1万とか大きいときに読み込みが非常に遅くなってしまうから。
        for i in range(min(30,len(dfTrain))):
            #1番目から順にdataにデータを格納していく
            data.append(list(dfTrain.iloc[i]))
        #chooseTarget.htmlを表示させる。カラム名と、上位30件のデータの内容をhtml側に渡して表示できるようにしている。
        return render_template("chooseTarget.html",columns=columns,data=data)
    #リクエストがPOSTの時の処理
    else:
        #送信されたデータから目的変数の名前を取得
        target = request.form["item"]
        #送信されたデータから分析の種類を取得
        analyticType = request.form["selection"]
        if analyticType == "分類":
            #analyticTypeをCにする。CはClassificationのC
            analyticType = "C"
            #訓練データの目的変数が100種類より多い場合、エラーを返す。分類の機械学習では分類するべき種類が多ければ多い時ほど学習に時間がかかり、この間はほかの人がサイトにアクセスしようとすると応答時間が極端に長くなってしまうので、制限を設けている
            #何種類まで許すかは適当に100にしているけど、この辺は一回みんなで話し合いたいかも
            if len(set(list(dfTrain[target]))) > 100:
                #種類が多すぎた場合、エラーページを表示させ、戻るボタンのリンク先をchooseTargetに設定する
                return render_template("error.html",error_message="この目的変数は分類には適しません",return_page=f"/{id}/chooseTarget")
        else:
            #選択された分析の種類が分類ではない時、analyticTypeをLにする
            #回帰：LegressionでLのつもりだったのですが、コメントアウト書きながら調べてみたら正しくはRegressionでした。(´・ω・｀)　今度直します
            analyticType = "L"
            #目的変数が数字であるかを判定している。回帰分析では目的変数は数字である必要があるので、文字などの場合はエラーページの飛ぶようにする。
            if dfTrain[target].dtype not in ["int64","float64"]:
                #エラーページを表示する
                return render_template("error.html", error_message="回帰分析では、目的変数を半角数字にしてください。", return_page=f"/{id}/chooseTarget")

        #分析の種類をidで紐づけて保存する
        analyticDict[id]=analyticType
        #目的変数をidで紐づけて保存する
        targetDict[id]=target
        #chooseColumnsのページに飛ばす
        return redirect(f"/{id}/chooseColumns")


#chooseColumnsページにアクセスされたときの処理
@app.route("/<string:id>/chooseColumns",methods=["GET","POST"])
def chooseColumns(id):
    #発行済みのid出なければホームページに戻す
    if id not in allowedID:
        return redirect("/")
    #idをキーとして学習に使うデータと、予測したいデータを持ってくる。
    dfTrain = dataDictTrain[id][1]
    dfTest = dataDictTest[id][1]
    #リクエストがGETの時の処理
    if request.method == "GET":
        #学習に使うデータのカラム名をcolumnsに代入
        columns = list(dfTrain.columns)
        #html側に渡すデータを作成する処理
        data = []
        for i in range(min(30,len(dfTrain))):
            data.append(list(dfTrain.iloc[i]))
        
        #chooseColuumn.htmlを表示させる。カラム名、上位30件のデータを表示できるように渡して、チェックを外させないために目的変数が何かも渡している。
        return render_template("chooseColumn.html",columns=columns,data=data,target=targetDict[id])
    else:
        #チェックボックスのチェックが入っている要素を取得する
        use_list=request.form.getlist("checkbox")
        #一応重複を許さないsetがたにしておく。また、この後 A not in B というBのなかにAがないか判定する処理があって、Bはただのリストよりset型にしておくと処理速度が格段に上がる。
        use_set=set(use_list)
        #使用するカラム名に目的変数のカラム名も追加。
        use_set.add(targetDict[id])
        #学習に使用するデータのカラム名を取得
        columns=list(dfTrain.columns)

        #すべてのカラム名に対して繰り返す。
        for column_name in columns:
            #もしあるカラム名が選んでもらった使用するカラム名一覧になかった場合、そのカラム名のものを学習に使用するデータと予測したいデータから削除する。
            if column_name not in use_set:
                dfTrain=dfTrain.drop(column_name,axis=1)
                dfTest=dfTest.drop(column_name,axis=1)
        #modelsで作ったclassをLGBMという変数に入れる
        LGBM = model_lightgbm()
        #分析の種類、目的変数、学習に使うデータ、予測したいデータを渡す
        #ここの内容はすべてmodels.pyに書かれていて、この一行で、データの前処理、学習用データをもとにモデルの作成、正解率やMAEといった指標の算出、重要度の算出など多くのことをやってくれる。
        LGBM.learning(analytic_type=analyticDict[id],target=targetDict[id],dfTrain = dfTrain,dfTest=dfTest)
        #変数ＬＧＢＭをidに紐づけて呼び出せるように、辞書型で保存
        modelDict[id] = [time.time(),LGBM]
        #scoreペーページに飛ばす
        return redirect(f"/{id}/score")

#scoreページにアクセスされたときの処理
@app.route("/<string:id>/score",methods=["GET","POST"])
def score(id):
    #発行していないidの場合はホームページに戻す
    if id not in allowedID:
        return redirect("/")

    if request.method == "GET":
        #idをキーとして、LGBM変数を読み込む
        LGBM = modelDict[id][1]
        #評価指標を分析の種類に応じて分けて取得する。
        if analyticDict[id] == "C":
            #分類では、分類の正解率を指標として使う。正解率の算出はすでに195行目のLGBM.learning()のところで行っているのでLGBM.accuracyとするだけで正解率を取得できるようになっている
            score = LGBM.accuracy
        else:
            #回帰では、MAE（Mean Absolute Error：平均絶対誤差）を指標として使う。MAEは簡単にいうと、平均したプラスマイナスの差
            score = LGBM.MAE
        #score.htmlを表示させる。accpctはscoreをパーセントに直すため100倍し、切り捨てたものにする。（四捨五入にすると実際の結果よりも少し高くなることがあるから切り捨てのほうが無難な気がした）impは算出した重要度のデータを入れている。
        return render_template("score.html",score = score, accpct = math.floor(score*100),imp=LGBM.imp,columns = LGBM.columns,id =id, analyticType=analyticDict[id])



@app.route("/<string:id>/predict",methods = ["GET","POST"])
def predict(id):
    #発行していないIDはホームページに戻す
    if id not in allowedID:
        return redirect("/")
    if request.method == "GET":
        #IDをキーにLGBMを取得
        LGBM = modelDict[id][1]
        #予測の実行。ここの処理もmodels.pyのなかに書かれていて、一行で予測結果の作成までできるようにしている
        LGBM.predict()

        #LGBMから予測結果のデータフレームを取得
        pred_df = LGBM.pred_df
        #予測結果のカラム名を取得
        columns = pred_df.columns
        #プレビューで表示する用のデータを入れるための変数を作成
        data = []
        #最大で、30件データを詰め込む
        for i in range(min(30,len(pred_df))):
            data.append(list(pred_df.iloc[i]))
        #result.htmlを表示、プレビュー用にカラム名と、上位30件のデータも一緒に渡す
        return render_template("result.html",id = id,columns=columns,data=data)


#ダウンロードページにアクセスされたときの処理
@app.route("/<string:id>/download",methods = ["GET","POST"])
def download(id):
    #発行していないIDはホームページに戻す
    if id not in allowedID:
        return redirect("/")
    #idをキーにLGBMを取得
    LGBM = modelDict[id][1]
    #予測結果を取得
    df = LGBM.pred_df
    #変数csvにcsvファイルの情報を代入させる。encodingでbom付きのutf8にしてるはずだけど、なぜかただのutf8になってたまに文字化けする問題が発生してる。index=FalseはFalseにしないとpandasの仕様で、データに1から順に番号が割り振られて、その番号も入った状態のCSVになってしまう
    csv = df.to_csv(encoding='utf-8-sig', index=False)

    #make_reponseはflaskの機能の一部で、ページにアクセスされたときの応答の内容を設定できる。今回はアクセスしたら自動でCSVファイルをダウンロードさせるようにしている
    response = make_response(csv)
    #ファイル名をdata.csvにする
    cd = 'attachment; filename=data.csv'
    response.headers['Content-Disposition'] = cd
    #ダウンロードさせるファイルの種類の設定 
    response.mimetype='text/csv'
    return response


#terms_of_useページにアクセスされたときの処理
@app.route("/terms_of_use",methods = ["GET","POST"])
def terms_of_use():
    if request.method == "GET":
        #terms_of_use.htmlを表示させる
        return render_template("terms_of_use.html")
    
#もしこのファイルが直接実行されている場合、app.runでサーバーを立ち上げる。このファイルを直接起動すると__name__には__main__が自動で入る
if __name__ == '__main__':
    app.run()