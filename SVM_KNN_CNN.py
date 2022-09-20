from cgi import test
import tkinter
from tkinter import *
from tkinter import filedialog
from tkinter import simpledialog
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
import os
import cv2
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, PrecisionRecallDisplay
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D

main = tkinter.Tk()
main.title("\t\t\t\t\t BTP B22PS01")
main.geometry("1300x1200")

global filename
global classifier
global svm_acc, knn_acc, cnn_acc
global X, Y
global X_train, X_test, y_train, y_test
global pca

def plotPRCurve(cls,label):
    display = PrecisionRecallDisplay.from_estimator(
    cls, X_test, y_test, name=label)
    _ = display.ax_.set_title("Precision-Recall curve")
    display.plot()

def uploadDataset():
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n")
    
def splitDataset():
    global X, Y
    global X_train, X_test, y_train, y_test
    global pca
    text.delete('1.0', END)
    X = np.load('features/X.txt.npy')
    Y = np.load('features/Y.txt.npy')
    X = np.reshape(X, (X.shape[0],(X.shape[1]*X.shape[2]*X.shape[3])))
    pca = PCA(n_components = 100)
    X = pca.fit_transform(X)
    #print(X.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Total CT Scan Images Found in dataset : "+str(len(X))+"\n")
    text.insert(END,"Train split dataset to 80% : "+str(len(X_train))+"\n")
    text.insert(END,"Test split dataset to 20%  : "+str(len(X_test))+"\n")

def tunning(model,parameters):
    clf = GridSearchCV(model,parameters)
    clf.fit(X_train,y_train)
    print("_"*139)
    print("\t\t\t\t\t\t SVM")
    print("_"*139)
    print("Best parameters for SVM: ",clf.best_params_)
    print(" ")
    return clf.best_params_

def executeSVM():
    global classifier #
    global svm_acc  #object
    text.delete('1.0', END) # For reading of test data (REading from top) or checking 
    cls = svm.SVC()  #classifier and svm is object and svc is class 
    params = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['linear', 'poly', 'rbf', 'sigmoid']}
    best_params = tunning(cls,params)
    cls.set_params(**best_params)
    cls.fit(X_train, y_train) #Training data pushing 
    predict = cls.predict(X_test) # predicitng x_test
    print("Confusion Matrix: ")
    print(confusion_matrix(y_test,predict))
    svm_acc = accuracy_score(y_test,predict) * 100 # Accuracy score
    classifier = cls #object 
    print(" ")
    print("SVM Accuracy: ", svm_acc)
    print(" ")
    print("_"*139)
    print("\t\t\t\t\t\t KNN")
    print("_"*139)
    text.insert(END,"SVM Accuracy : "+str(svm_acc)+"\n") # Output
    plotPRCurve(cls,"RBF SVM")

def KfoldCrossValidation(model):
    pipeline = make_pipeline(model)
    strtfdKFold = StratifiedKFold(n_splits=10)
    kfold = strtfdKFold.split(X_train, y_train)
    scores = []
    for k, (train, test) in enumerate(kfold):
        pipeline.fit(X_train[train, :], y_train[train])
        score = pipeline.score(X_train[test, :], y_train[test])
        scores.append(score)
        
        print('Fold: %2d, Training/Test Split Distribution: %s, Accuracy: %.3f' % (k+1, np.bincount(y_train[train]), score))
    print('\nCross-Validation accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores)))

def executeKNN():
    global knn_acc
    cls = KNeighborsClassifier(n_neighbors = 2)
    print("K-fold Cross Validation for 10 splits")
    KfoldCrossValidation(cls) 
    cls.fit(X_train, y_train)
    predict = cls.predict(X_test)
    print("\nConfusion Matrix")
    print(confusion_matrix(y_test,predict))
    knn_acc= accuracy_score(y_test,predict) * 100
    text.insert(END,"KNN Accuracy : "+str(knn_acc)+"\n")
    print("\nKnn Accuracy:", knn_acc)
    plotPRCurve(cls,"KNN")

def executeCNN():
    global cnn_acc
    X = np.load('features/X.txt.npy')
    Y = np.load('features/Y.txt.npy')
    Y = to_categorical(Y)
    classifier = Sequential()
    classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (2, 2)))
    classifier.add(Flatten())
    classifier.add(Dense(units = 256, activation = 'relu'))
    classifier.add(Dense(units = 2, activation = 'softmax'))
    print(classifier.summary())
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    hist = classifier.fit(X, Y, batch_size=16, epochs=12, shuffle=True, verbose=2)
    hist = hist.history
    acc = hist['accuracy']
    cnn_acc = acc[9] * 100
    text.insert(END,"CNN Accuracy : "+str(cnn_acc)+"\n")
    print("\nCNN Accuracy :", cnn_acc)

def predictCancer():
    filename = filedialog.askopenfilename(initialdir="testSamples")
    img = cv2.imread(filename)
    img = cv2.resize(img, (64,64))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(64,64,3)
    im2arr = im2arr.astype('float32')
    im2arr = im2arr/255
    test = []
    test.append(im2arr)
    test = np.asarray(test)
    test = np.reshape(test, (test.shape[0],(test.shape[1]*test.shape[2]*test.shape[3])))
    test = pca.transform(test)
    predict = classifier.predict(test)[0]
    msg = ''
    if predict == 0:
        msg = "Uploaded CT Scan is Normal"
    if predict == 1:
        msg = "Uploaded CT Scan is Abnormal"
    img = cv2.imread(filename)
    img = cv2.resize(img, (400,400))
    cv2.putText(img, msg, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 255), 2)
    cv2.imshow(msg, img)
    cv2.waitKey(0)    

def graph():
    height = [svm_acc,knn_acc, cnn_acc]
    bars = ('SVM Accuracy','KNN Accuracy','CNN Accuracy')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()

font = ('times', 14, 'bold')
title = Label(main, text='Prediction of time-to-event outcomes in diagnosing lung cancer based on SVM, KNN and CNN algorithm')
title.config(bg='black', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=150)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=50,y=120)
text.config(font=font1)


font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Lung Cancer Dataset", command=uploadDataset)
uploadButton.place(x=50,y=550)
uploadButton.config(font=font1)  

readButton = Button(main, text="Read & Split Dataset to Train & Test", command=splitDataset)
readButton.place(x=300,y=550)
readButton.config(font=font1) 

svmButton = Button(main, text="Execute SVM Accuracy", command=executeSVM)
svmButton.place(x=50,y=625)
svmButton.config(font=font1)

predictButton = Button(main, text="Predict Lung Cancer", command=predictCancer)
predictButton.place(x=50,y=700)
predictButton.config(font=font1)

kmeansButton = Button(main, text="Execute KNN Accuracy", command=executeKNN)
kmeansButton.place(x=300,y=625)
kmeansButton.config(font=font1) 

meansButton = Button(main, text="Execute CNN Accuracy Algorithm", command=executeCNN)
meansButton.place(x=500,y=625)
meansButton.config(font=font1) 

graphButton = Button(main, text="Accuracy Graph", command=graph)
graphButton.place(x=300,y=700)
graphButton.config(font=font1) 

main.config(bg='#B2BEB5')
main.mainloop()
