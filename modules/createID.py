#ランダムな値を生成するライブラリ
import random
#何かの文字列を、全く違う文字列へと変換（ハッシュ化）をするライブラリ
import hashlib
#時間に関するライブラリ
import time

#IDに使う文字
string = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstu"

#IDを生成する関数
def getID():
    #現在の時間を取得し、取得した数字データを文字データとして扱い、ハッシュ化をしてランダムな文字を作る。
    #時間をもとにハッシュ化をする理由は、時間ごとに生成されるIDを必ず違うものになり、IDの重複を防げるから。
    result = hashlib.sha256(str(time.time()).encode("utf-8")).hexdigest()

    #万が一完ぺきに同じタイミングでIDを生成すると上の、時間をもとにしたハッシュ化は同じ文字列を生成してしまうので、10文字のランダムな文字列をさらに加える
    for i in range(10):
        #IDに上で作ったstring変数のなかからランダムな文字を選び、追加する
        result += string[random.randint(0,len(string)-1)]
    #生成したIDを返す
    return result
