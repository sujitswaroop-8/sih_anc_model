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

