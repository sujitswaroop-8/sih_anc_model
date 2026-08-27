#file paths
from pathlib import Path
import numpy as np
import librosa
import numpy as np
fan_folder=Path.home() /"Documents"/"fan_noise"
fan_noise=[]
for file in fan_folder.iterdir():
    fan_noise.append(file)

speech_folder=Path.home() /"Documents"/"Speech audio"
D={"train":[],"test":[],"dev":[]}
for folder in speech_folder.iterdir():
    if folder.is_dir():
        if (folder.name=="dev_new"):
            D["dev"]=list(folder.glob("*.flac"))
        if (folder.name=="test_new"):
            D["test"]=list(folder.glob("*.flac"))
        if (folder.name=="train_new"):
            D["train"]=list(folder.glob("*.flac"))

#function to extract 28 features (MFCC, RMS ENERGY, ZERO CROSSING RATE)
def extract_features(file):
    #LOAD ONE AUDIO FILE
    audio,sr=librosa.load(file,sr=None,mono=True)

    #Extract 13 MFCCs
    mfcc=librosa.feature.mfcc(y=audio,sr=sr,n_mfcc=13)
    #average each mfcc across time
    mfcc_mean=np.mean(mfcc,axis=1)

    #variation of each mfcc across time (standard deviation)
    mfcc_std=np.std(mfcc,axis=1)
    

    #RMS energy across time (taking avg to get one feature)
    rms=librosa.feature.rms(y=audio)
    rms_mean=np.mean(rms)
    
    #ZERO CORSSING RATE (ZCR) (across time, take abg to get 1 feature)
    zcr=librosa.feature.zero_crossing_rate(audio)
    zcr_mean=np.mean(zcr)
    
    features = np.concatenate([
        mfcc_mean,
        mfcc_std,
        [rms_mean],
        [zcr_mean]
    ])
    
    return features


    
    #load the files in & and extract features

#Speech training files
speech_train=[]
for i in range(5000):
    features=extract_features(D["train"][i])
    speech_train.append(features)
print(len(speech_train))


               
#Fan Noise training files
fan_train=[]
for i in range(350):
    features=extract_features(fan_noise[i])
    fan_train.append(features)

speech_label=1
noise_label=0
X_train=np.vstack([speech_train,fan_train]) #vertical stack 
y_train=np.concatenate([[1]*len(speech_train),[0]*len(fan_train)])


speech_label=1
noise_label=0
X_train=np.vstack([speech_train,fan_train]) #vertical stack 

y_train=np.concatenate([[1]*len(speech_train),[0]*len(fan_train)])

#now shuffle training data
from sklearn.utils import shuffle
X_train,y_train=shuffle(X_train,y_train,random_state=42) #the mapping of X and y is not lost as they both get the same shuffle
y_train

#FEATURE SCALING
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
scaler = StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
plt.scatter(
    X_train_scaled[:, 0], #first feature, and all the rows (audio)
    X_train_scaled[:, 1], #second feature,and all the rows (audio)
    c=y_train, #one colour for speech, the other for noise
    alpha=0.5 #50% transparency
)

#Train the Logistic Regression model
from sklearn.linear_model import LogisticRegression
model=LogisticRegression(class_weight="balanced",max_iter=1000)
model.fit(X_train_scaled,y_train)


#dev set (validation)
speech_dev=[]
for i in range (500):
    features=extract_features(D["dev"][i])
    speech_dev.append(features)
fan_dev=[]
for i in range(280,340):
    features=extract_features(fan_noise[i])
    fan_dev.append(features)

X_dev=np.vstack([speech_dev,fan_dev])
y_dev=np.concatenate([[1]*len(speech_dev),[0]*len(fan_dev)])
#Scale X_dev
X_dev_scaled=scaler.transform(X_dev) #use the statistics learned from the scaling of training data

#make predictions
y_pred=model.predict(X_dev_scaled)
print(y_pred)
print(y_dev)

#TESTING
speech_test=[]
for i in range (500):
    features=extract_features(D["test"][i])
    speech_test.append(features)
fan_test=[]
for i in range(340,400):
    features=extract_features(fan_noise[i])
    fan_test.append(features)
#scaling the test data
X_test=np.vstack([speech_test,fan_test])
y_test=np.concatenate([[1]*len(speech_test),[0]*len(fan_test)])
#Scale X_test
X_test_scaled=scaler.transform(X_test) #use the statistics learned from the scaling of training data


#make predictions
y_pred=model.predict(X_test_scaled)
print(y_pred)
print(y_test)

#metrics
from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
score = model.score(X_test_scaled, y_test)
print(score)

