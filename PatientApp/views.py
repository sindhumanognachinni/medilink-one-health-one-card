from django.shortcuts import render
import pymysql
from datetime import datetime

from django.template import RequestContext
from django.contrib import messages
import pymysql
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
import os
from keras.models import load_model
import matplotlib.pyplot as plt #use to visualize dataset vallues
import io
import base64
import cv2
import numpy as np
from keras.models import Sequential, load_model, Model
import pickle
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.decomposition import PCA
from keras.utils.np_utils import to_categorical

global username, doctor

labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
accuracy = []
precision = []
recall = []
fscore = []

#function to calculate accuracy and other metrics
def calculateMetrics(algorithm, predict, y_test):
    global accuracy, precision, recall, fscore
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)    

X = np.load('model/X.txt.npy')
Y = np.load('model/Y.txt.npy')
#preprocess images like shuffling and normalization
X = X.astype('float32')
X = X/255 #normalized pixel values between 0 and 1
indices = np.arange(X.shape[0])
np.random.shuffle(indices)#shuffle all images
X = X[indices]
Y = Y[indices]
Y = to_categorical(Y)
#split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
cnn_model = load_model("model/cnn_weights.hdf5")

X_train1 = np.reshape(X_train, (X_train.shape[0], (X_train.shape[1] * X_train.shape[2] * X_train.shape[3])))  
X_test1 = np.reshape(X_test, (X_test.shape[0], (X_test.shape[1] * X_test.shape[2] * X_test.shape[3])))  
y_test1 = np.argmax(y_test, axis=1)
y_train1 = np.argmax(y_train, axis=1)
X_train1 = X_train1[0:1000]
y_train1 = y_train1[0:1000]
#apply PCA to select features
pca = PCA(n_components=300)
X_train1 = pca.fit_transform(X_train1)
X_test1 = pca.transform(X_test1)
#train SVM algorithm
svm_cls = svm.SVC()
svm_cls.fit(X_train1, y_train1)
predict = svm_cls.predict(X_test1)
calculateMetrics("SVM", predict, y_test1)

#perform prediction on test data   
predict = cnn_model.predict(X_test)
predict = np.argmax(predict, axis=1)
y_test1 = np.argmax(y_test, axis=1)
#call this function to calculate accuracy and other metrics
calculateMetrics("CNN", predict, y_test1)

#use this function to predict fish species uisng extension model
def predict(image_path, cnn_model):
    labels = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']
    image = cv2.imread(image_path)#read test image
    img = cv2.resize(image, (32,32))#resize image
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,32,32,3)#convert image as 4 dimension
    img = np.asarray(im2arr)
    img = img.astype('float32')#convert image features as float
    img = img/255 #normalized image
    predict = cnn_model.predict(img)#now predict dog breed
    predict = np.argmax(predict)
    img = cv2.imread(image_path)
    img = cv2.resize(img, (400,300))#display image with predicted output
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.putText(img, 'Predicted As : '+labels[predict], (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
    return img, labels[predict]   

def UploadMRIAction(request):
    if request.method == 'POST':
        global uname
        today = str(datetime.now())
        patient = request.POST.get('t2', False)
        cnn_model = load_model("model/cnn_weights.hdf5")
        myfile = request.FILES['t1'].read()
        fname = request.FILES['t1'].name
        with open("PatientApp/static/reports/"+patient+".jpg", "wb") as file:
            file.write(myfile)
        file.close()
        img, disease = predict("PatientApp/static/reports/"+patient+".jpg", cnn_model)
        dbconnection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'doctorpatientapp',charset='utf8')
        dbcursor = dbconnection.cursor()
        qry = "INSERT INTO mri(patient_name,mri_file,detected_disease,entry_date) VALUES('"+str(patient)+"','"+fname+"','"+disease+"','"+today+"')"
        dbcursor.execute(qry)
        dbconnection.commit()
        plt.imshow(img)
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()
        context= {'data': img_b64}
        return render(request, 'UploadMRI.html', context)   
    

def AppointmentAction(request):
    if request.method == 'POST':
        global username
        doctor = request.POST.get('t1', False)
        disease = request.POST.get('t2', False)
        dd = request.POST.get('dd', False)
        mm = request.POST.get('mm', False)
        yy = request.POST.get('yy', False)
        date = yy+"-"+mm+"-"+dd
        today = str(datetime.now())
        bid = 0
        mysqlConnect = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'doctorpatientapp',charset='utf8')
        with mysqlConnect:
            result = mysqlConnect.cursor()
            result.execute("select max(appointment_id) from appointment")
            lists = result.fetchall()
            for ls in lists:
                bid = ls[0]
        if bid != None:
            bid = bid + 1
        else:
            bid = 1
        dbconnection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'doctorpatientapp',charset='utf8')
        dbcursor = dbconnection.cursor()
        qry = "INSERT INTO appointment(appointment_id,patient_name,doctor_name,disease_details,prescription,appointment_date,booking_date) VALUES('"+str(bid)+"','"+username+"','"+doctor+"','"+disease+"','Pending','"+date+"','"+str(today)+"')"
        dbcursor.execute(qry)
        dbconnection.commit()
        if dbcursor.rowcount == 1:
            data = "Your Appointment Confirmed on "+date
            context= {'data':data}
            return render(request,'PatientScreen.html', context)
        else:
            data = "Error in making appointment"
            context= {'data':data}
            return render(request,'PatientScreen.html', context)     
            

def Appointment(request):
    if request.method == 'GET':
        global doctor
        doctor = request.GET['doctor']
        today = datetime.now()
        month = today.month
        year = today.year
        day = today.day
        print(str(month)+" "+str(year)+" "+str(day))
        output = '<tr><td><font size="3" color="black">Doctor</td><td><input type="text" name="t1" size="25" value="'+doctor+'" readonly/></td></tr>'
        output += '<tr><td><font size="3" color="black">Appointment&nbsp;Date</td><td><select name="dd">'
        for i in range(day,32):
            if i <= 9:
                output += '<option value="0'+str(i)+'">0'+str(i)+'</option>'
            else:
                output += '<option value="'+str(i)+'">'+str(i)+'</option>'
        output += '</select><select name="mm">'
        for i in range(month,13):
            if i <= 9:
                output += '<option value="0'+str(i)+'">0'+str(i)+'</option>'
            else:
                output += '<option value="'+str(i)+'">'+str(i)+'</option>'
        output += '</select><select name="yy">'
        for i in range(year,(year+5)):
            output += '<option value="'+str(i)+'">'+str(i)+'</option>'
        output += '</select>'
        context= {'data':output}
        return render(request,'BookAppointment.html', context)

def getDetails(user):
    address = ""
    email = ""
    phone = ""
    mysqlConnect = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'doctorpatientapp',charset='utf8')
    with mysqlConnect:
        result = mysqlConnect.cursor()
        result.execute("select phone_no, email, address from user_signup where username='"+user+"'")
        lists = result.fetchall()
        for ls in lists:
            phone = ls[0]
            email = ls[1]
            address = ls[2]
    return phone, email, address        
    

def ViewPrescription(request):
    if request.method == 'GET':
        global username
        output = '<table border=1 align=center>'
        output+='<tr><th><font size=3 color=black>Appointment ID</font></th>'
        output+='<th><font size=3 color=black>Patient Name</font></th>'
        output+='<th><font size=3 color=black>Doctor Name</font></th>'
        output+='<th><font size=3 color=black>Disease Details</font></th>'
        output+='<th><font size=3 color=black>Prescription</font></th>'
        output+='<th><font size=3 color=black>Appointment Date</font></th>'
        output+='<th><font size=3 color=black>Booking Date</font></th></tr>'
        mysqlConnect = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'doctorpatientapp',charset='utf8')
        with mysqlConnect:
            result = mysqlConnect.cursor()
            result.execute("select * from appointment where patient_name='"+username+"'")
            lists = result.fetchall()
            for ls in lists:
                output+='<tr><td><font size=3 color=black>'+str(ls[0])+'</font></td>'
                output+='<td><font size=3 color=black>'+ls[1]+'</font></td>'
                output+='<td><font size=3 color=black>'+ls[2]+'</font></td>'
                output+='<td><font size=3 color=black>'+ls[3]+'</font></td>'
                output+='<td><font size=3 color=black>'+ls[4]+'</font></td>'
                output+='<td><font size=3 color=black>'+str(ls[5])+'</font></td>'
                output+='<td><font size=3 color=black>'+ls[6]+'</font></td></tr>'
        context= {'data':output}        
        return render(request,'PatientScreen.html', context) 

def GeneratePrescription(request):
    if request.method == 'GET':
        global username
        bid = request.GET['pid']
        output = '<tr><td><font size="3" color="black">Appointment&nbsp;ID</td><td><input type="text" name="t1" size="25" value="'+bid+'" readonly/></td></tr>'
        context= {'data':output}
        return render(request,'GeneratePrescription.html', context)

def GeneratePrescriptionAction(request):
    if request.method == 'POST':
        bid = request.POST.get('t1', False)
        prescription = request.POST.get('t2', False)
        dbconnection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'doctorpatientapp',charset='utf8')
        dbcursor = dbconnection.cursor()
        qry = "update appointment set prescription='"+prescription+"' where appointment_id='"+bid+"'"
        dbcursor.execute(qry)
        dbconnection.commit()
        if dbcursor.rowcount == 1:
            data = "Prescription Updated Successfully"
            context= {'data':data}
            return render(request,'DoctorScreen.html', context)
        else:
            data = "Error in adding prescription details"
            context= {'data':data}
            return render(request,'DoctorScreen.html', context) 
    

def ViewAppointments(request):
    if request.method == 'GET':
        global username
        today = datetime.now()
        month = str(today.month)
        year = str(today.year)
        day = str(today.day)
        date = str(year)+"-"+str(month)+"-"+str(day)
        print(date)
        output = '<table border=1 align=center>'
        output+='<tr><th><font size=3 color=black>Appointment ID</font></th>'
        output+='<th><font size=3 color=black>Patient Name</font></th>'
        output+='<th><font size=3 color=black>Doctor Name</font></th>'
        output+='<th><font size=3 color=black>Disease Details</font></th>'
        output+='<th><font size=3 color=black>Prescription</font></th>'
        output+='<th><font size=3 color=black>Appointment Date</font></th>'
        output+='<th><font size=3 color=black>Booking Date</font></th>'
        output+='<th><font size=3 color=black>Generate Description</font></th></tr>'
        mysqlConnect = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'doctorpatientapp',charset='utf8')
        with mysqlConnect:
            result = mysqlConnect.cursor()
            result.execute("select * from appointment where doctor_name='"+username+"' and appointment_date='"+date+"'")
            lists = result.fetchall()
            for ls in lists:
                output+='<tr><td><font size=3 color=black>'+str(ls[0])+'</font></td>'
                output+='<td><font size=3 color=black>'+ls[1]+'</font></td>'
                output+='<td><font size=3 color=black>'+ls[2]+'</font></td>'
                output+='<td><font size=3 color=black>'+ls[3]+'</font></td>'
                output+='<td><font size=3 color=black>'+ls[4]+'</font></td>'
                output+='<td><font size=3 color=black>'+str(ls[5])+'</font></td>'
                output+='<td><font size=3 color=black>'+ls[6]+'</font></td>'
                if ls[4] == 'Pending':
                    output+='<td><a href=\'GeneratePrescription?pid='+str(ls[0])+'\'><font size=3 color=black>Click Here for Prescription</font></a></td></tr>'
                else:
                    output+='<td><font size=3 color=black>Already Generated Prescription</font></td>'
        context= {'data':output}            
        return render(request,'DoctorScreen.html', context)

def BookAppointment(request):
    if request.method == 'GET':
        output = '<table border=1 align=center>'
        output+='<tr><th><font size=3 color=black>Doctor Name</font></th>'
        output+='<th><font size=3 color=black>Phone No</font></th>'
        output+='<th><font size=3 color=black>Email ID</font></th>'
        output+='<th><font size=3 color=black>Address</font></th>'
        output+='<th><font size=3 color=black>Description</font></th>'
        output+='<th><font size=3 color=black>Book Appointment</font></th></tr>'
        mysqlConnect = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'doctorpatientapp',charset='utf8')
        with mysqlConnect:
            result = mysqlConnect.cursor()
            result.execute("select username,phone_no,email,address,description from user_signup where usertype='Doctor'")
            lists = result.fetchall()
            for ls in lists:
                output+='<tr><td><font size=3 color=black>'+ls[0]+'</font></td>'
                output+='<td><font size=3 color=black>'+ls[1]+'</font></td>'
                output+='<td><font size=3 color=black>'+ls[2]+'</font></td>'
                output+='<td><font size=3 color=black>'+ls[3]+'</font></td>'
                output+='<td><font size=3 color=black>'+ls[4]+'</font></td>'
                output+='<td><a href=\'Appointment?doctor='+ls[0]+'\'><font size=3 color=black>Click Here to Book Appointment</font></a></td></tr>'
        context= {'data':output}        
        return render(request,'PatientScreen.html', context)    

def index(request):
    if request.method == 'GET':
        global accuracy, precision, recall, fscore
        output = '<table border=1 align=center>'
        output+='<tr><th><font size=3 color=black>Algorithm Name</font></th>'
        output+='<th><font size=3 color=black>Accuracy</font></th>'
        output+='<th><font size=3 color=black>Precision</font></th>'
        output+='<th><font size=3 color=black>Recall</font></th>'
        output+='<th><font size=3 color=black>FSCORE</font></th></tr>'
        output+='<tr><td><font size=3 color=black>SVM</font></td>'
        output+='<td><font size=3 color=black>'+str(accuracy[0])+'</font></td>'
        output+='<td><font size=3 color=black>'+str(precision[0])+'</font></td>'
        output+='<td><font size=3 color=black>'+str(recall[0])+'</font></td>'
        output+='<td><font size=3 color=black>'+str(fscore[0])+'</font></td></tr>'
        output+='<tr><td><font size=3 color=black>CNN</font></td>'
        output+='<td><font size=3 color=black>'+str(accuracy[1])+'</font></td>'
        output+='<td><font size=3 color=black>'+str(precision[1])+'</font></td>'
        output+='<td><font size=3 color=black>'+str(recall[1])+'</font></td>'
        output+='<td><font size=3 color=black>'+str(fscore[1])+'</font></td></tr></table><br/><br/><br/><br/>'
        context= {'data':output}        
        return render(request,'index.html', context)

def UploadMRI(request):
    if request.method == 'GET':
       return render(request, 'UploadMRI.html', {})        

def Register(request):
    if request.method == 'GET':
       return render(request, 'Register.html', {})
    
def DoctorLogin(request):
    if request.method == 'GET':
       return render(request, 'DoctorLogin.html', {})

def PatientLogin(request):
    if request.method == 'GET':
       return render(request, 'PatientLogin.html', {})

def isUserExists(username):
    is_user_exists = False
    global details
    mysqlConnect = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'doctorpatientapp',charset='utf8')
    with mysqlConnect:
        result = mysqlConnect.cursor()
        result.execute("select * from user_signup where username='"+username+"'")
        lists = result.fetchall()
        for ls in lists:
            is_user_exists = True
    return is_user_exists    

def RegisterAction(request):
    if request.method == 'POST':
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        contact = request.POST.get('t3', False)
        email = request.POST.get('t4', False)
        address = request.POST.get('t5', False)
        desc = request.POST.get('t6', False)
        usertype = request.POST.get('t7', False)
        record = isUserExists(username)
        page = None
        if record == False:
            dbconnection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'doctorpatientapp',charset='utf8')
            dbcursor = dbconnection.cursor()
            qry = "INSERT INTO user_signup(username,password,phone_no,email,address,description,usertype) VALUES('"+str(username)+"','"+password+"','"+contact+"','"+email+"','"+address+"','"+desc+"','"+usertype+"')"
            dbcursor.execute(qry)
            dbconnection.commit()
            if dbcursor.rowcount == 1:
                data = "Signup Done! You can login now"
                context= {'data':data}
                return render(request,'Register.html', context)
            else:
                data = "Error in signup process"
                context= {'data':data}
                return render(request,'Register.html', context) 
        else:
            data = "Given "+username+" already exists"
            context= {'data':data}
            return render(request,'Register.html', context)


def checkUser(uname, password, utype):
    global username
    msg = "Invalid Login Details"
    mysqlConnect = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = '', database = 'doctorpatientapp',charset='utf8')
    with mysqlConnect:
        result = mysqlConnect.cursor()
        result.execute("select * from user_signup where username='"+uname+"' and password='"+password+"' and usertype='"+utype+"'")
        lists = result.fetchall()
        for ls in lists:
            msg = "success"
            username = uname
            break
    return msg

def PatientLoginAction(request):
    if request.method == 'POST':
        global username
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        msg = checkUser(username, password, "Patient")
        if msg == "success":
            context= {'data':"Welcome "+username}
            return render(request,'PatientScreen.html', context)
        else:
            context= {'data':msg}
            return render(request,'PatientLogin.html', context)
        
def DoctorLoginAction(request):
    if request.method == 'POST':
        global username
        username = request.POST.get('t1', False)
        password = request.POST.get('t2', False)
        msg = checkUser(username, password, "Doctor")
        if msg == "success":
            context= {'data':"Welcome "+username}
            return render(request,'DoctorScreen.html', context)
        else:
            context= {'data':msg}
            return render(request,'DoctorLogin.html', context)










        


        
