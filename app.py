from flask import Flask,render_template,request
import pickle
import os
import numpy as np


app = Flask("__name__")
model = pickle.load (open("D:\VELOCITY_Class\practice datasets\\bank\\artifacts\\bank_model.pkl","rb"))
columns = pickle.load (open("d:\VELOCITY_Class\practice datasets\\bank\\artifacts\\columns.pkl","rb"))


@app.route("/")
def index():
    return render_template("index.html")



@app.route("/predict",methods=["POST","GET"])
def get_credit():

    data = request.form
    vector = np.zeros(26)

    job = data['job'] 
    job_list = columns.get('columns')[15:].tolist()
    job_index = job_list.index(job)
    vector[job_index] == 1


    var_age=float(request.form.get("age"))
    vector[0]=var_age

    var_marital=int(request.form.get("marital"))
    vector[1]=var_marital

    var_education=int(request.form.get("education"))
    vector[2]=var_education

    var_default=int(request.form.get("default"))
    vector[3]=var_default

    var_balance=float(request.form.get("balance"))
    vector[4]=var_balance

    var_housing=int(request.form.get("housing"))
    vector[5]=var_housing

    var_loan=int(request.form.get("loan"))
    vector[6]=var_loan

    var_contact=int(request.form.get("contact"))
    vector[7]=var_contact

    var_day=int(request.form.get("day"))
    vector[8]=var_day

    var_month=int(request.form.get("month"))
    vector[9]=var_month

    var_duration=float(request.form.get("duration"))
    vector[10]=var_duration

    var_campaign=float(request.form.get("campaign"))
    vector[11]=var_campaign

    var_pdays=float(request.form.get("pdays"))
    vector[12]=var_pdays

    var_previous=float(request.form.get("previous"))
    vector[13]=var_previous

    var_poutcome=int(request.form.get("poutcome"))
    vector[14]=var_poutcome

    input = [vector]
    print(vector)
 
    result=model.predict(input)  # 2d array
    print(result)

    if result[0]==1:
        outp="client subscribed a term deposit"
        return render_template("index.html",prediction=outp)
    else:
        outp="client not subscribed a term deposit"
        return render_template("index.html",prediction=outp)



if __name__=="__main__":
    app.run(debug=True,host='127.0.0.1',port=5000)

