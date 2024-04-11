#勾配ブースティング決定木とよばれる計算速度が速くて精度も高いモデルを作れるライブラリ
import lightgbm as lgb
#学習データをモデルの作成に使用する訓練用のデータと、実際にどれくらいの精度なのかをテストするための検証用データに分割できる関数
from sklearn.model_selection import train_test_split
#CSVのようなデータフレイムを扱うライブラリ
import pandas as pd
#高速で様々な数学的な演算を行えるライブラリ
import numpy as np
#機械学習のライブラリのsklearnから、行列にかかわる内容をインポート
from sklearn import metrics
#ラベルエンコ⁻ディングを行うためのもの（エンコーディングは、機械学習に使えない文字データを数値データに変換することで、ラベルエンコーディングは特に文字に0から準に数字を割り当てて変換する方法 例:[りんご、みかん、りんご、ぶどう、みかん] -> [0,1,0,2,1]）
from sklearn.preprocessing import LabelEncoder
#モデルの精度を評価するための指標を計算する関数。mean absolute errorは平均絶対誤差で、回帰分析のときに平均でプラスマイナスどれくらいのずれがあるのかを計算できる
from sklearn.metrics import mean_absolute_error


#lightgbmのクラスを作成する。クラスの中にはいくつも関数を作ることができる。特に self.変数名 のようにselfを付けた変数は同じクラス内ならほかの関数でも使うことができるので便利
class model_lightgbm:
    #学習する関数の定義、入力として分析の方法、目的変数、学習用データ、予測したいデータを受け取る。最初の引数がselfになっているのは、クラスの中の関数の第一引数はselfにするという決まりがあるから
    def learning(self,analytic_type,target,dfTrain,dfTest):

        #学習用データと予測したいデータを一つにくっつけたデータを作成する。ラベルエンコーディングするときに、学習用と予測したいものを別々でラベルエンコーディングすると割り当てられる数字が変わってしまうから、くっつけたデータに対してラベルエンコーディングをする必要がある
        dfAll = pd.concat([dfTrain,dfTest])


        #analytic_typeをself.analytic_typeにすることで、このlearning関数だけでなく下のpredict関数でも使えるようにする
        self.analytic_type=analytic_type
        #目的変数もselfを付けてほかの関数でも使えるようにする
        self.target=target
        #ラベルエンコーディングして数値に変換した後に、最後にもう一度元の文字データに戻せるように、カラム名と変換内容を紐づけて保存する辞書型を作る
        self.labelencoders = {}
        #目的変数に対して使うラベルエンコーディングをself.le_targetとする。leはラベルエンコーディングの略で、機械学習系のコードでよく見かける気がする
        self.le_target = LabelEncoder()

        
        #もし分析方法が分類(Classification)だったとき
        if self.analytic_type == "C":
            #目的変数のデータをどのように変換するかを計算する。　（目的変数=self.le_target.fit(目的変数) のようにすると、どの文字データを何の数字にするか決めたうえでその値で上書きされるが、「目的変数＝」を付けないとどの文字を何の数字に変換するか決めただけでまだデータ本体は返還されていない）
            self.le_target.fit(dfAll[self.target])

            #上でどの文字を何の数字にするか決めたものをもとに、学習用データにラベルエンコーディングを適用して変換する
            dfTrain[self.target] = self.le_target.transform(dfTrain[self.target])
            #もし、予測用データに目的変数が含まれている場合、ラベルエンコーディングする。（基本的には未知のデータを予測するためにツールを使用することがおおそうだから、予測用データには目的変数が含まれないことのほうが多そう）
            if self.target in dfTest.columns:
                #予測用データの目的変数にラベルエンコーディングを適用して変換する
                dfTest[self.target] = self.le_target.transform(dfTest[self.target])
        
        #すべてのカラムに対して、データが文字型なのかを判定し、文字型ならラベルエンコーディングで数値に変換する
        for colum in (dfAll.columns):
            #もし、あるカラムのデータタイプが整数（int64）,少数(float64)でないかつ、そのカラムが目的変数でない場合、ラベルエンコーディングをする
            if dfAll[colum].dtype not in ["int64","float64"] and colum != self.target:
                
                #変数leにLabelEncoderの機能を入れる（インスタンス化）
                le=LabelEncoder()
                #あるカラムのどの文字を何の数字に決めるかを決定する
                le.fit(dfAll[colum])
                #学習用データと訓練用データの欠損値をNoneという文字で置き換える
                dfTrain[colum].fillna("None")
                dfTest[colum].fillna("None")
                
                #訓練用データと予測したいデータのあるカラムについて　le.fit(dfAll[colum]) で決定した変換方法に従って文字を数値へと変換する
                dfTrain[colum]=le.transform(dfTrain[colum])
                dfTest[colum] = le.transform(dfTest[colum])
                #どのように変換したかを保存して、後々数値にしたのをもとの文字に戻せるようにしておく
                #ラベルエンコーディングをしたカラム名と変換方法などを決めたleを紐づけて保存する
                self.labelencoders[colum] = le

        #もし、分析方法が回帰であった場合、目的変数が欠損値になっているものは学習に使えないので、そのような行を削除する
        if self.analytic_type == "L":
            #学習用データと予測したいデータを合わせたものから目的変数が欠損地ではないもののみを残す
            dfAll = dfAll[dfAll[self.target].notnull()]
            #学習用データから目的変数が欠損地でないもののみ残す
            dfTrain = dfTrain[dfTrain[self.target].notnull()]
        
        #dfTrainをself.trainDataにすることでほかの関数でも使えるようにする
        self.trainData = dfTrain
        #dfTestをself.testDataにすることでほかの関数でも使えるようにする
        self.testData = dfTest

        #変数yを学習用データの目的変数にする
        y = dfTrain[target]
        #変数xを全体のデータから目的変数をなくしたもの、つまり説明変数にする。
        x = dfTrain.drop([target], axis=1)

        #学習に使用する訓練データとモデルの精度を評価する評価用のデータに分割する。訓練用8：評価用2で分割する
        #x_trainが学習に使う目的変数、x_testが評価に使う目的変数、y_trainが学習に使う目的変数、y_testが評価に使う目的変数
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)

        #学習用と評価用でそれぞれ説明変数と目的変数をセットにしたデータセットを作成する
        lgb_train = lgb.Dataset(x_train, y_train)
        lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)

        #もし分析の種類が分類であった時の機械学習のコードを書く
        if analytic_type == "C":
            #パラメータを作成する。GCIのタイタニックの時のコードをもとにしているが、objective : binaly　だと2値分類（生存か死亡か、男性か女性かなど）しかできないので、多値分類（ミカンかブドウかそれともイチゴか など3種類以上あるとき）もできるようにmulticlassに変えてある。multiclassでも二値分類は可能
            params = {
                'task': 'train',
                'boosting_type': 'gbdt',
                'objective': 'multiclass', # binaly:二値分類, multiclass:多クラス分類
                #multiclassをobjectiveに指定した場合、目的変数の種類が何種類なのかを指定する必要がある。今回は目的変数が入ったyをset型にして、その長さを渡している。setがたは重複を許さないlistのようなもので、例えば[みかん、ぶどう、りんご、りんご、みかん]をsetがたにすると[みかん,ぶどう,りんご]のように重複を消してくれる。そしてset型で重複を消した要素の個数が分類の種類の個数になる
                "num_class":len(set(y)),
                #GCIの時はaucだったがauc_muに変えてある。aucだと二値分類にしか適用できないが、auc_muは3種類以上の多値分類にも適用できる。aucの意味は正解率であり、metricに正解率を指定するとできる限り、予測の正解率を高めようという方針で機械学習が進んでいく
                'metric': {'auc_mu'},

                #ここから下のパラメーターはタイタニックに特化したパラメーターで、逆にタイタニック以外のデータを入れたときに精度を下げてしまう要因になっていたので、いったんコメントアウトしてある
                #'num_leaves': 20,
                #'max_depth':5,
                #'min_data_in_leaf': 3,
                #'num_iteration': 100,
                #'learning_rate':0.03,
                #'num_boost_round':100,
                #'early_stopping_rounds':20,
            }

            #lightgbmの学習を行う処理、先ほど設定したパラメータと、訓練データ、検証用データを渡してある
            self.model = lgb.train(
                params,
                train_set=lgb_train, # トレーニングデータの指定
                valid_sets=lgb_eval, # 検証データの指定
                        )
            
            
            #検証用説明変数をもとに、予測を行い、その結果をy_predに入れる
            y_pred = self.model.predict(x_test)

            

            #そのクラスに属する確率が返ってくるから、最も高い確率のクラスに加工する
            y_pred = np.argmax(y_pred,axis=1)
            #accuracy_scoreという正解率を求める関数をつかって、y_predと検証用目的変数のy_testとの正解率を計算する
            self.accuracy = metrics.accuracy_score(y_test,y_pred )
            
            
        
        else:
            #ここからは回帰の処理を書く
            #パラメータの設定、rmseはRoot Mean Square Errorの略で、予測と正解の差を二乗したものを平均して、最後にルートをとったもの。回帰分析での王道の指標で、metricにrmseを指定することで、このモデルはrmseの指標を最小化するように学習を進める
            params = {'metric' : 'rmse'}
            #パラメータと学習用のデータセットを渡して学習を行う
            self.model = lgb.train(params,lgb_train)

            #検証用説明変数を使って、予測を行いその結果をy_predに入れる
            y_pred = self.model.predict(x_test)
            
            #精度の表示に使うMAEという指標を計算する。MAEはMean Absolute Errorのことで、予測と正解の差に絶対値をとって、それを平均したもの。プラスマイナスだいたいどれくらいのずれがあるかわかる。RMSEのほうがよくつかわれるが、機械学習初心者の人がこのツールを使うことを想定するとMAEのほうが直感的に理解し安いと思ってMAEにしてある
            self.MAE = mean_absolute_error(y_test,y_pred)
            
        #各説明変数の重要度をリスト型にしてimp(importanceの略のつもり)に入れる
        imp = list(self.model.feature_importance())
        #重要度は例えば[300,50,6]のように足していくつになるといった規則がないので、わかりやすいように全体の何パーセントの重要度なのかに変換する。[300,50,6]だったら[300/(300+50+6)、50/(300+50+6)、6/(300+50+6)]のようにして、四捨五入する
        imp = list(map(lambda x: round(x/sum(imp) * 100),imp))
        #説明変数のすべてのカラム名をリスト型にして、columnsに入れる
        columns = list(x.columns)

        #重要度の割合が入ったimpと、カラム名が入ったcolumnsをペアにしてdataという変数に入れる
        data = []
        for i,c in zip(imp,columns):#zipは、まずimpの1番目をiに、columnsの1番目をcに入れ、次にimpの2番目をiに、columnsの2番目をcに入れ、次は3番目をそれぞれiとcにいれ、・・・のようにして繰り返される
            data.append([i,c]) #重要度の割合のiと、カラム名のcをセットにして、dataに追加していく
        
        #scoreページで重要度を表示するとき、重要度が大きい順に並べたいので、昇順にしておく。並び替える要素が[重要度,カラム名]のようにただの数字じゃなくてリスト型のようなときは、それぞれのリスト型の一番最初の要素（今回だと重要度）をもとに並び替えが行われる。
        #何も指定しないとsorted関数は小さい順に並べてしまうので、reverse=Trueにして大きい順に並べるようにしている。
        data = sorted(data,reverse=True)

        #self.impに上で大きい順にした重要度の値をいれる。selfを付けると、ほかの関数で読み出せれる以外に、このクラスを　変数名=クラス名()　として呼び出したとき、 変数名.imp　でこのself.impを呼び出せるようになるメリットがある。実際にmain.pyファイルでLGBMという変数名を使って、LGBM.impとLGBM.columnsでこの重要度を参照し、score.htmlにそのデータを渡して重要度のグラフを描画している
        self.imp = []
        #上で大きい順にしたカラム名を入れるリスト型を作る
        self.columns = []
        #dataの要素をひとつづつdに入れて繰り返す
        for d in data:
            #dataのある要素の1番目をself.impに追加する　[重要度、カラム名]なので重要度が選ばれる。　pythonでは一番目を0、二番目を1、三番目を2のようにして指定する。
            self.imp.append(d[0])
            #dataのある要素の2番目をself.columnsに追加する　[重要度、カラム名]なのでカラム名が選ばれる
            self.columns.append(d[1])




    #予測したいデータに対して、モデルを適用し、予測値を出す関数。予測値を出すには、本来なら学習したモデルや、予測したいデータを渡す必要があるが上のlearnign関数内で、self.model、self.testDataのようにselfを付けてあるので、この関数内でもself.modelやself.testDataで学習済みのモデルや予測したいデータを読み込める
    #selfが入っているのはクラス内の関数には必ず第一引数をselfにするという決まりから入れている
    def predict(self):
        #上のlearnign関数で作ったself.testDataを読み込み、pred_dfという変数に代入する
        pred_df = self.testData

        #もし、予測したいデータの中に目的変数が含まれているならば、予測したいデータから目的変数を削除する
        if self.target in pred_df.columns:
            pred_df = pred_df.drop([self.target], axis=1)
        
        #learning関数ですでに学習済みのself.modelを使って、予測したいデータのpred_dfに対して予測を行う
        pred = self.model.predict(pred_df)

        #もし分析の種類が分類なら
        if self.analytic_type == "C":
            #predにはそれぞれの確立（例えばブドウが80%、リンゴが10％、みかんが10%で[0.8,0.1,0.1]のような感じ）になっているから、最も確率が高いのが何かを求める
            pred = np.argmax(pred,axis=1)
        
        #予測したいデータに新しく目的変数名をカラム名とした列を追加し、データは予測値にする
        pred_df[self.target] = pred

        #もし分析の種類が分類なら、ラベルエンコーディングで文字型は数値型へと変換してあるので、目的変数に対しての変換方法を定めたself.le_targetにたいし、inverse_transformをして、元に戻す
        if self.analytic_type == "C":
            pred_df[self.target] = self.le_target.inverse_transform(pred_df[self.target])

        #説明変数も文字データをラベルエンコーディングで数値にしたままだと、ダウンロードしたときによくわからない状態になっているので、文字データに復元させる。
        #カラム名と変換方法をself.labelencodersという辞書型に紐づけて保存してあるので、1ペアづつ読み込んで文字データの復元をする
        for colum in self.labelencoders.keys():
            #self.labelencodersに保存したあるカラム名に対し、それの変換方法をself.labelencoders[カラム名]で呼び出し、.inverse_transformで復元する
            pred_df[colum] = self.labelencoders[colum].inverse_transform(pred_df[colum])

        #予測値をくっつけて、文字を数値に変えていたものをもとに戻したデータフレイムをself.pred_dfとしてselfを付けることで、ほかのところから呼び出せるようにしておく。
        #今回はmain.pyのほうで、LGBMという変数に、model_lightgbmのクラスを代入（インスタンス化）していて、main.pyでダウンロードページにアクセスされたときの処理のところで、df = LGBM.pred_df　でこのデータフレイムを呼び出している
        self.pred_df = pred_df
        







