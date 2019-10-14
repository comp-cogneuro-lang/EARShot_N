'''
This script requires three libraries: librosa, numpy, scipy
If there is an error, please setup the libraries.
To install, run the following command in the terminal.

sudo pip install numpy, scipy, librosa

'''
import librosa, os;
import numpy as np;
import scipy.io;

frequency = 22050;  #The frequency of wave file
maxv = np.iinfo(np.int16).max   #32767
wav_Folder = "D:/Simulation_Raw_Data/EARS/SAY_VOICES_TIMIT_LEX";    #Change the folder that there is the wave files


for root, directory_List, file_Name_List in os.walk(wav_Folder):
    if not os.path.exists(root.replace(wav_Folder, wav_Folder + "_Trim")):  #When there is no export folder, make new one.
        os.makedirs(root.replace(wav_Folder, wav_Folder + "_Trim"));
    for file_Name in file_Name_List:
        if not file_Name.lower().endswith('.wav'):  #Only wav file is trimmed.
            continue;
        sig = librosa.core.load(os.path.join(root, file_Name).replace("\\", "/"), sr = frequency)[0]; #Signal load
        trimmed_Sig = librosa.effects.trim(sig, frame_length = 32, hop_length=16)[0]    #Delete the silence of front and back
        scipy.io.wavfile.write(os.path.join(root.replace(wav_Folder, wav_Folder + "_Trim"), file_Name).replace("\\", "/"), rate = frequency, data=(trimmed_Sig * maxv).astype(np.int16));  #Signal save