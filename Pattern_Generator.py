import scipy.io.wavfile as wav
import os, io, librosa, gzip
import numpy as np
import _pickle as pickle
from random import shuffle
from scipy import signal
from scipy.io import loadmat
try: from gensim.models import KeyedVectors
except: pass

from Audio import *
from Hyper_Parameters import pattern_Parameters

#Global constants
with open(pattern_Parameters.Word_List_File, "r") as f:
    readLines = f.readlines()
    pronunciation_Dict = {word.upper(): pronunciation.split(".") for word, pronunciation in [x.strip().split("\t") for x in readLines]}
    using_Word_List = list(pronunciation_Dict.keys())

if pattern_Parameters.Pattern_Use_Bit == 16:
    pattern_Bit_Type = np.float16
elif pattern_Parameters.Pattern_Use_Bit == 32:
    pattern_Bit_Type = np.float32
else:
    assert False
    
if pattern_Parameters.Semantic_Mode.upper() == "SRV".upper():
    semantic_Indice_Dict = {}
    for word in using_Word_List:
        unit_List = list(range(pattern_Parameters.SRV.Size))
        while True:
            shuffle(unit_List)
            if not set(unit_List[0:pattern_Parameters.SRV.Assign_Number]) in semantic_Indice_Dict.values():
                semantic_Indice_Dict[word] = set(unit_List[0:pattern_Parameters.SRV.Assign_Number])
                break

    semantic_Dict = {}
    for word, index_Set in semantic_Indice_Dict.items():
        new_Semantic_Pattern = np.zeros(shape= (pattern_Parameters.SRV.Size), dtype= pattern_Bit_Type)
        for unit_Index in index_Set:
            new_Semantic_Pattern[unit_Index] = 1
        semantic_Dict[word] = new_Semantic_Pattern

elif pattern_Parameters.Semantic_Mode.upper() == "Word2Vec".upper():
    semantic_Dict = KeyedVectors.load_word2vec_format(pattern_Parameters.Word2Vec.DB_File_Path, binary=True)
    
if not os.path.exists(pattern_Parameters.Pattern_Path):
    os.makedirs(pattern_Parameters.Pattern_Path)


def Pattern_File_Geneate(
    word,
    pronunciation,
    identifier, #In paper, this is 'talker'.
    voice_File_Path,
    ):
    new_Pattern_Dict = {"Word": word, "Pronunciation": pronunciation, "Identifier": identifier}
    
    if pattern_Parameters.Acoutsic_Mode.upper() == "Spectrogram".upper():
        sig = librosa.core.load(voice_File_Path, sr = pattern_Parameters.Spectrogram.Sample_Rate)[0]
        sig = librosa.effects.trim(sig, frame_length = 32, hop_length=16)[0]  #Trim
        spec = spectrogram(
            sig,
            num_freq= pattern_Parameters.Spectrogram.Dimension,
            frame_shift_ms= pattern_Parameters.Spectrogram.Frame_Shift,
            frame_length_ms= pattern_Parameters.Spectrogram.Frame_Length,
            sample_rate= pattern_Parameters.Spectrogram.Sample_Rate,
            )
        new_Pattern_Dict["Acoustic"] = np.transpose(spec).astype(pattern_Bit_Type)
    elif pattern_Parameters.Acoutsic_Mode.upper() == "Mel".upper():
        sig = librosa.core.load(voice_File_Path, sr = pattern_Parameters.Mel.Sample_Rate)[0]
        sig = librosa.effects.trim(sig, frame_length = 32, hop_length=16)[0]  #Trim
        mel_Spec = melspectrogram(
            sig,
            num_freq= pattern_Parameters.Mel.Spectrogram_Dim,
            frame_shift_ms= pattern_Parameters.Mel.Frame_Shift,
            frame_length_ms= pattern_Parameters.Mel.Frame_Length,
            num_mels= pattern_Parameters.Mel.Mel_Dim,
            sample_rate= pattern_Parameters.Mel.Sample_Rate,
            max_abs_value= pattern_Parameters.Mel.Max_Abs
            )
        new_Pattern_Dict["Acoustic"] = np.transpose(mel_Spec).astype(pattern_Bit_Type)
    else:
        assert False

    if pattern_Parameters.Semantic_Mode.upper() == "SRV".upper():
        new_Pattern_Dict["Semantic"] = semantic_Dict[word].astype(pattern_Bit_Type)
    elif pattern_Parameters.Semantic_Mode.upper() == "Word2Vec".upper():
        new_Pattern_Dict["Semantic"] = semantic_Dict[word].astype(pattern_Bit_Type)
    else:
        assert False
    new_Pattern_Dict["Is_Sentence"] = False    #Now unsupported
    
    pattern_File_Name = os.path.split(voice_File_Path)[1].replace(os.path.splitext(voice_File_Path)[1], ".pickle").upper()
    
    with open(os.path.join(pattern_Parameters.Pattern_Path, pattern_File_Name).replace("\\", "/"), "wb") as f:
        pickle.dump(new_Pattern_Dict, f, protocol= 2)

    print("{}\t->\t{}".format(voice_File_Path, pattern_File_Name))
    

def Metadata_Generate():
    new_Metadata_Dict = {}

    #Although we use the hyper parameter now, I insert several information about that for checking consistency.
    new_Metadata_Dict["Hyper_Parameter_Dict"] = {        
        "Pattern_Use_Bit": pattern_Parameters.Pattern_Use_Bit,
        "Acoutsic_Mode": pattern_Parameters.Acoutsic_Mode,
        "Semantic_Mode": pattern_Parameters.Semantic_Mode,
        }

    if pattern_Parameters.Acoutsic_Mode.upper() == "Spectrogram".upper():
        new_Metadata_Dict["Acoustic_Size"] = pattern_Parameters.Spectrogram.Dimension
        new_Metadata_Dict["Hyper_Parameter_Dict"]["Spectrogram_Info"] = {
            "Sample_Rate": pattern_Parameters.Spectrogram.Sample_Rate,
            "Frame_Length": pattern_Parameters.Spectrogram.Frame_Length,
            "Frame_Shift": pattern_Parameters.Spectrogram.Frame_Shift,
            }
    elif pattern_Parameters.Acoutsic_Mode.upper() == "Mel".upper():
        new_Metadata_Dict["Acoustic_Size"] = pattern_Parameters.Mel.Mel_Dim
        new_Metadata_Dict["Hyper_Parameter_Dict"]["Mel_Info"] = {
            "Sample_Rate": pattern_Parameters.Mel.Sample_Rate,
            "Frame_Length": pattern_Parameters.Mel.Frame_Length,
            "Frame_Shift": pattern_Parameters.Mel.Frame_Shift,            
            'Spectrogram_Dim': pattern_Parameters.Mel.Spectrogram_Dim,        
            'Max_Abs': pattern_Parameters.Mel.Max_Abs,
            }
    elif pattern_Parameters.Acoutsic_Mode.upper() == "Cochleagram".upper():
        new_Metadata_Dict["Acoustic_Size"] = 47
        new_Metadata_Dict["Hyper_Parameter_Dict"]["Cochleagram_Info"] = {
            "Window": pattern_Parameters.Cochleagram.Window,
            }
    elif pattern_Parameters.Acoutsic_Mode.upper() == "Neurogram".upper():
        new_Metadata_Dict["Acoustic_Size"] = 50
        new_Metadata_Dict["Hyper_Parameter_Dict"]["Cochleagram_Info"] = {
            "Window": pattern_Parameters.Cochleagram.Window,
            }
    else:
        assert False

    if pattern_Parameters.Semantic_Mode.upper() == "SRV".upper():
        new_Metadata_Dict["Semantic_Size"] = pattern_Parameters.SRV.Size
        new_Metadata_Dict["Hyper_Parameter_Dict"]["SRV_Info"] = {
            "Assign_Number": pattern_Parameters.SRV.Assign_Number,
            }
    elif pattern_Parameters.Semantic_Mode.upper() == "Word2Vec".upper():
        new_Metadata_Dict["Semantic_Size"] = pattern_Parameters.Word2Vec.Size
    else:
        assert False
    new_Metadata_Dict["Pronunciation_Dict"] = pronunciation_Dict   #key: word, value: pronunciation
    new_Metadata_Dict["Pattern_Path_Dict"] = {}    #key: (word, identifier), value: pattern_Path
    new_Metadata_Dict["Word_and_Identifier_Dict"] = {}     #key: pattern_Path, value: (word, identifier) #Reversed of pattern_Path_Dict
    new_Metadata_Dict["Cycle_Dict"] = {}     #key: pattern_Path, value: pattern cycle
    for root, dirs, files in os.walk(pattern_Parameters.Pattern_Path):
        for file in files:
            if file.upper() == "Metadata.pickle".upper():
                continue
            with open(os.path.join(root, file).replace("\\", "/"), "rb") as f:
                pattern_Dict = pickle.load(f)
            new_Metadata_Dict["Pattern_Path_Dict"][pattern_Dict["Word"], pattern_Dict["Identifier"]] = file
            new_Metadata_Dict["Word_and_Identifier_Dict"][file] = (pattern_Dict["Word"], pattern_Dict["Identifier"])
            new_Metadata_Dict["Cycle_Dict"][file] = pattern_Dict["Acoustic"].shape[0]            

    new_Metadata_Dict["Target_Dict"] ={word: semantic_Dict[word] for word in using_Word_List}

    with open(os.path.join(pattern_Parameters.Pattern_Path, "METADATA.PICKLE").replace("\\", "/"), "wb") as f:
        pickle.dump(new_Metadata_Dict, f, protocol= 2)


def Metadata_Subset_Generate(word_List = None, identifier_List = None, metadata_File_Name = "METADATA.SUBSET.PICKLE"):
    if not word_List is None:
        word_List = [x.upper() for x in word_List]
    if not identifier_List is None:
        identifier_List = [x.upper() for x in identifier_List]    

    with open(os.path.join(pattern_Parameters.Pattern_Path, "METADATA.PICKLE").replace("\\", "/"), "rb") as f:
        metadata_Dict = pickle.load(f)

    new_Metadata_Dict = {}
    new_Metadata_Dict["Hyper_Parameter_Dict"] = metadata_Dict["Hyper_Parameter_Dict"]
    new_Metadata_Dict["Acoustic_Size"] = metadata_Dict["Acoustic_Size"]
    new_Metadata_Dict["Semantic_Size"] = metadata_Dict["Semantic_Size"]
        
    if not word_List is None:
        word_Filtered_Pattern_Path_List = [pattern_Path for pattern_Path, (word, identifier) in metadata_Dict["Word_and_Identifier_Dict"].items() if word in word_List]
    else:
        word_Filtered_Pattern_Path_List = [pattern_Path for pattern_Path, (word, identifier) in metadata_Dict["Word_and_Identifier_Dict"].items()]
    if not identifier_List is None:
        identifier_Filtered_Pattern_Path_List = [pattern_Path for pattern_Path, (word, identifier) in metadata_Dict["Word_and_Identifier_Dict"].items() if identifier in identifier_List]
    else:
        identifier_Filtered_Pattern_Path_List = [pattern_Path for pattern_Path, (word, identifier) in metadata_Dict["Word_and_Identifier_Dict"].items()]
    
    new_Metadata_Dict["Pronunciation_Dict"] = metadata_Dict["Pronunciation_Dict"]

    new_Metadata_Dict["Pattern_Path_Dict"] = {
        (word, identifier): pattern_Path
        for (word, identifier), pattern_Path in metadata_Dict["Pattern_Path_Dict"].items()
        if pattern_Path in word_Filtered_Pattern_Path_List and pattern_Path in identifier_Filtered_Pattern_Path_List
        }

    new_Metadata_Dict["Word_and_Identifier_Dict"] = {
        pattern_Path: (word, identifier)
        for pattern_Path, (word, identifier) in metadata_Dict["Word_and_Identifier_Dict"].items()
        if pattern_Path in word_Filtered_Pattern_Path_List and pattern_Path in identifier_Filtered_Pattern_Path_List
        }

    new_Metadata_Dict["Cycle_Dict"] = {
        pattern_Path: cycle
        for pattern_Path, cycle in metadata_Dict["Cycle_Dict"].items()
        if pattern_Path in word_Filtered_Pattern_Path_List and pattern_Path in identifier_Filtered_Pattern_Path_List
        }

    if not word_List is None:
        new_Metadata_Dict["Target_Dict"] = {
            word: target_Pattern
            for word, target_Pattern in metadata_Dict["Target_Dict"].items()
            if word in word_List
            }
    else:
        new_Metadata_Dict["Target_Dict"] = metadata_Dict["Target_Dict"]
    
    with open(os.path.join(pattern_Parameters.Pattern_Path, metadata_File_Name).replace("\\", "/"), "wb") as f:
        pickle.dump(new_Metadata_Dict, f, protocol= 2)

if __name__ == '__main__':
    with open('Pronunciation_Data_1K.txt', 'r') as f:
        word_List = [x.strip().split('\t')[0].strip() for x in f.readlines()]

    Metadata_Subset_Generate(
        word_List = word_List,
        #identifier_List= ["Agnes", "Alex", 'Allison', 'Ava',  "Bruce", "Fred", "Junior", "Kathy", "Princess", "Ralph", 'Samantha', 'Susan', 'Tom', "Vicki", "Victoria"],
        metadata_File_Name = "METADATA.1KW.17T.PICKLE"
        )