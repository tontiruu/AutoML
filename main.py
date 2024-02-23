from flask import Flask,render_template,redirect,request,url_for,session
import pandas as pd
from modules import createID



app = Flask(__name__,static_folder="static")
global data_dict

@app.route("/",methods=["POST","GET"])
def index():
    id = createID.getID()
    if request.method == "GET":
        return render_template("index.html",id=id)


@app.route("/home/<string:id>",methods=["GET","POST"])
def home(id):
    if request.method == "GET":
        return render_template("home.html")
    else:
        print(request.files)
        file = request.files["CSVfile"]
        file.save("test.csv")
        return redirect(f"/choseTarget/{id}")
    


if __name__ == '__main__':
    app.run()