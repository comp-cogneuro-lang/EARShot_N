import numpy as np
import tensorflow as tf
import _pickle as pickle
import time, os, sys, ctypes, zipfile, shutil, argparse
from EARShot import EARShot_Model as Model
from Customized_Functions import Correlation2D, Batch_Correlation2D, Cosine_Similarity2D, Batch_Cosine_Similarity2D, MDS, Z_Score, Wilcoxon_Rank_Sum_Test2D, Mean_Squared_Error2D, Euclidean_Distance2D
from Hyper_Parameters import pattern_Parameters, model_Parameters

tf_Session = tf.Session()   # Session is a manager of Tensorflow. This session is for PSI and FSI calculation

#Talker list
talker_List = ["Agnes", "Alex", "Bruce", "Fred", "Junior", "Kathy", "Princess", "Ralph", "Vicki", "Victoria"]   #Now 10 talkers. If some talkers are added later, add to here.
talker_List = [x.upper() for x in talker_List]  #Capitalizing of talker

#Feature load
with open("Phoneme_Feature.txt", "r", encoding='UTF8') as f:    #Load features of each phoneme
    readLines = f.readlines()

feature_List = readLines[0].strip().split("\t")[3:] #Feature name list
index_feature_Name_Dict = {index: feature_Name.strip() for index, feature_Name in enumerate(feature_List)}  # Index to feature name matching
feature_Name_Index_Dict = {feature_Name.strip(): index for index, feature_Name in index_feature_Name_Dict.items()}  # Feature name to Index matching

#Phoneme list and feature dict
phoneme_Label_Dict = {} #key, value: CMU code, IPA
consonant_List = []
vowel_List = []
feature_Dict = {feature_Name: [] for feature_Name in index_feature_Name_Dict.values()}

for readLine in readLines[1:]:
    raw_Data = readLine.strip().split("\t")
    phoneme_Label_Dict[raw_Data[0]] = raw_Data[1]   #CMU code to IPA matching
    
    #Checking consonant or vowel
    if raw_Data[2] == "1":
        consonant_List.append(raw_Data[0])
    elif raw_Data[2] == "0":
        vowel_List.append(raw_Data[0])
    
    # Checking each features have which phoneme
    for feature_Name_Index, value in enumerate([int(feature.strip()) for feature in raw_Data[3:]]):
        if value == 1:
            feature_Dict[index_feature_Name_Dict[feature_Name_Index]].append(raw_Data[0])

phoneme_List = consonant_List + vowel_List

def Export_Alignment_List_Dict_by_Single_Phone(alignment_Path):
    with open(alignment_Path, 'r') as f:
        lines = f.readlines()
    
    alignment_Dict = {phoneme: [] for phoneme in phoneme_List}
    alignment_Dict.update({(phoneme, talker): [] for phoneme in phoneme_List for talker in talker_List})
    for line in lines[1:]:
        word, talker, phoneme, xMin, xMax = line.strip().split('\t')
        alignment_Dict[phoneme].append((word, talker, float(xMin), float(xMax)))
        alignment_Dict[phoneme, talker].append((word, talker, float(xMin), float(xMax)))

    return alignment_Dict

def Export_Alignment_List_Dict_by_Feature(alignment_Path):
    with open(alignment_Path, 'r') as f:
        lines = f.readlines()
    
    alignment_Dict = {feature: [] for feature in feature_List}
    alignment_Dict.update({(feature, talker): [] for feature in feature_List for talker in talker_List})
    for line in lines[1:]:
        word, talker, phoneme, xMin, xMax = line.strip().split('\t')
        for feature, feature_Phoneme_List in feature_Dict.items():
            if phoneme in feature_Phoneme_List:
                alignment_Dict[feature].append((word, talker, float(xMin), float(xMax)))
                alignment_Dict[feature, talker].append((word, talker, float(xMin), float(xMax)))
        
    return alignment_Dict

def Phoneme_Feature_Compatibility_Cheker(alignment_Path):
    for key, alignment_List in Export_Alignment_List_Dict_by_Single_Phone(alignment_Path).items():
        if len(alignment_List) == 0:                        
            raise ValueError('There is no reference about the phoneme \'{}\'.'.format(key))
    for key, alignment_List in Export_Alignment_List_Dict_by_Feature(alignment_Path).items():
        if len(alignment_List) == 0:                        
            raise ValueError('There is no reference about the feature \'{}\'.'.format(key))

def Activation_Dict_Generate(
    model,
    voice_Path= 'D:/Pattern/EARShot/WAVS_ONLY_Padded',
    alignment_Path= 'Alignment_Data.txt',
    is_Absolute = True,
    batch_Size=1000
    ):
    #Generating pattern path list
    voice_File_Path_List = []    
    for root, _, file_Name_List in os.walk(voice_Path):
        for filename in file_Name_List:
            name, ext = os.path.splitext(filename.upper())
            if not ext == '.WAV':
                continue
            word, talker = name.split('_')
            voice_File_Path_List.append(os.path.join(root, filename).replace("\\", "/"))

    word_Talker_Index_Dict = {}
    for index, filename in enumerate(voice_File_Path_List):
        word, talker = os.path.splitext(os.path.basename(filename.upper()))[0].split('_')
        word_Talker_Index_Dict[word, talker] = index

    activation_Tensor = model.hidden_Plot_Tensor_List[0]  #[Batch, Hidden, Time]
    if is_Absolute:
        activation_Tensor = tf.abs(activation_Tensor)   #Applying absolute to 'Hidden activation'.
    
    model.tf_Session.run(model.test_Mode_Turn_On_Tensor_List) #Backup the hidden state. Initial hidden state is zero vector.       

    activation_List = []
    for batch_Index in range(0, len(voice_File_Path_List), batch_Size):        
        activation = model.tf_Session.run(  #In this line, model calucate the hidden activation.
            fetches = activation_Tensor,
            feed_dict = model.pattern_Feeder.Get_Test_Pattern_from_Voice(voice_File_Path_List=voice_File_Path_List[batch_Index:batch_Index+batch_Size])
            )    #[Mini_Batch, Hidden, Time]
        activation_List.append(activation)
    
    model.tf_Session.run(model.test_Mode_Turn_Off_Tensor_List)     #Restore the hidden state (not necessary)

    max_Activation_Length = max([x.shape[2] for x in activation_List])
    activation = np.vstack([
        np.concatenate([x, np.zeros((x.shape[0], x.shape[1], max_Activation_Length- x.shape[2])) * np.nan], axis= 2)
        for x in activation_List
        ])  # [Mini_Batch, Hidden, Time]
    
    max_Step_Length = 0
    with open(alignment_Path, 'r') as f:
        lines = f.readlines()
    for line in lines[1:]:
        line = line.strip().split('\t')
        max_Step_Length = max(max_Step_Length, int(float(line[4]) * 100 - float(line[3]) * 100))

    activation_Dict_by_Single_Phone = {}
    for key, alignment_List in Export_Alignment_List_Dict_by_Single_Phone(alignment_Path).items():  #key: phoneme or (phoneme, talker), value: alignment info list
        activation_Dict_by_Single_Phone[key] = []
        for word, talker, xMin, xMax in alignment_List:
            #The reason of '* 100': The unit of alignment is second, but model's step is 10ms.
            index = word_Talker_Index_Dict[word, talker]
            activation_Slice = activation[index][:, int(xMin * 100):int(xMax * 100)]
            activation_Slice = np.concatenate(
                [
                    activation_Slice,   # [Hidden, Time]
                    np.zeros([
                        activation_Slice.shape[0],
                        max_Step_Length - activation_Slice.shape[1]
                        ]) * np.nan
                    ],
                axis = 1
                )   #Nan Padding
            activation_Dict_by_Single_Phone[key].append(activation_Slice)
        activation_Dict_by_Single_Phone[key] = np.stack(activation_Dict_by_Single_Phone[key], axis= 0)  #[Sample, Hidden, Time]
        
    activation_Dict_by_Feature = {}
    for key, alignment_List in Export_Alignment_List_Dict_by_Feature(alignment_Path).items():  #key: phoneme or (phoneme, talker), value: alignment info list
        activation_Dict_by_Feature[key] = []
        for word, talker, xMin, xMax in alignment_List:
            #The reason of '* 100': The unit of alignment is second, but model's step is 10ms.
            index = word_Talker_Index_Dict[word, talker]
            activation_Slice = activation[index][:, int(xMin * 100):int(xMax * 100)]
            activation_Slice = np.concatenate(
                [
                    activation_Slice,    # [Hidden, Time]
                    np.zeros([
                        activation_Slice.shape[0],
                        max_Step_Length - activation_Slice.shape[1]                        
                        ]) * np.nan
                    ],
                axis = 1
                )   #Nan Padding
            activation_Dict_by_Feature[key].append(activation_Slice)
        activation_Dict_by_Feature[key] = np.stack(activation_Dict_by_Feature[key], axis= 0)  #[Sample, Hidden, Time]

    return activation_Dict_by_Single_Phone, activation_Dict_by_Feature

def PSI_Dict_Generate(hidden_Size, activation_Dict_by_Single_Phone, criterion_List):
    #For PSI. The activation is averaged by phoneme and time steps, so the flow information disappear.
    avg_Activation_Dict = {}
    avg_Activation_Consonant = np.stack([np.nanmean(activation_Dict_by_Single_Phone[consonant], axis=(0,2)) for consonant in consonant_List], axis = 1) #[Unit, Consonant], Mean and consonant stacking
    avg_Activation_Vowel = np.stack([np.nanmean(activation_Dict_by_Single_Phone[vowel], axis=(0,2)) for vowel in vowel_List], axis = 1) #[Unit, Vowel], Mean and vowel stacking
    avg_Activation_Dict["All"] = np.hstack([avg_Activation_Consonant, avg_Activation_Vowel])   #[Unit, Phoneme], Stacking both of consonant and vowel. 
    for talker in talker_List:  #If user want talker specific data.
        avg_Activation_Consonant = np.stack([np.nanmean(activation_Dict_by_Single_Phone[consonant, talker], axis=(0,2)) for consonant in consonant_List], axis = 1) #[Unit, Consonant]
        avg_Activation_Vowel = np.stack([np.nanmean(activation_Dict_by_Single_Phone[vowel, talker], axis=(0,2)) for vowel in vowel_List], axis = 1) #[Unit, Vowel]
        avg_Activation_Dict[talker] = np.hstack([avg_Activation_Consonant, avg_Activation_Vowel])   #[Unit, Phoneme]
        
    #PSI Dict
    avg_Activation_Placeholder = tf.placeholder(tf.float32, shape=(None,)) #[Phoneme]
    criterion_Placeholder = tf.placeholder(tf.float32)

    tiled_Sample = tf.tile(tf.expand_dims(avg_Activation_Placeholder, axis=1), multiples=[1, tf.shape(avg_Activation_Placeholder)[0]])  #The activation array is tiled for 2D calculation.
    
    #Over criterion, getting 1 point.
    positive_Significant_Map_Tensor = tf.sign(tf.clip_by_value(tiled_Sample - (tf.transpose(tiled_Sample) + criterion_Placeholder), 0, 1)) #[Phoneme, Phoneme], Comparing each other phonemes by positive direction (Negative becomes 0).
    negative_Significant_Map_Tensor = tf.sign(tf.clip_by_value(tf.transpose(tiled_Sample) - (tiled_Sample + criterion_Placeholder), 0, 1)) #[Phoneme, Phoneme], Comparing each other phonemes by negative direction (Positive becomes 0).

    positive_PSI_Map_Tensor = tf.reduce_sum(positive_Significant_Map_Tensor, axis=1)    #[Phoneme], Sum score
    negative_PSI_Map_Tensor = tf.reduce_sum(negative_Significant_Map_Tensor, axis=1)    #[Phoneme], Sum score

    psi_Dict = {}
    for talker in ["All"] + talker_List:
        for criterion in criterion_List:
            for direction, map_Tensor in [("Positive", positive_PSI_Map_Tensor), ("Negative", negative_PSI_Map_Tensor)]:
                psi_Dict[criterion, direction, talker] = np.stack([
                    tf_Session.run( #Conducting above tensor.
                        fetches= map_Tensor,
                        feed_dict = {
                            avg_Activation_Placeholder: avg_Activation_Dict[talker][unit_Index],
                            criterion_Placeholder: criterion
                            }
                        ) for unit_Index in range(hidden_Size)],
                    axis=1
                    )

    return psi_Dict

def FSI_Dict_Generate(hidden_Size, activation_Dict_by_Feature, criterion_List):
    #For FSI. The activation is averaged by feature and time steps, so the flow information disappear.
    avg_Activation_Dict = {}
    avg_Activation_Dict["All"] = np.stack([np.nanmean(activation_Dict_by_Feature[feature], axis=(0,2)) for feature in feature_List], axis = 1) #[Unit, Feature], Mean and feature stacking
    for talker in talker_List:  #If user want talker specific data.
        avg_Activation_Dict[talker] = np.stack([np.nanmean(activation_Dict_by_Feature[feature, talker], axis=(0,2)) for feature in feature_List], axis = 1) #[Unit, Feature]


    #FSI Dict
    avg_Activation_Placeholder = tf.placeholder(tf.float32, shape=(None,)) #[Feature]
    criterion_Placeholder = tf.placeholder(tf.float32)

    tiled_Sample = tf.tile(tf.expand_dims(avg_Activation_Placeholder, axis=1), multiples=[1, tf.shape(avg_Activation_Placeholder)[0]])  #The activation array is tiled for 2D calculation.
    
    #Over criterion, getting 1 point.
    positive_Significant_Map_Tensor = tf.sign(tf.clip_by_value(tiled_Sample - (tf.transpose(tiled_Sample) + criterion_Placeholder), 0, 1)) #[Feature, Feature], Comparing each other phonemes by positive direction (Negative becomes 0).
    negative_Significant_Map_Tensor = tf.sign(tf.clip_by_value(tf.transpose(tiled_Sample) - (tiled_Sample + criterion_Placeholder), 0, 1)) #[Feature, Feature], Comparing each other phonemes by negative direction (Positive becomes 0).

    positive_FSI_Map_Tensor = tf.reduce_sum(positive_Significant_Map_Tensor, axis=1)    #[Feature], Sum score
    negative_FSI_Map_Tensor = tf.reduce_sum(negative_Significant_Map_Tensor, axis=1)    #[Feature], Sum score

    fsi_Dict = {}
    for talker in ["All"] + talker_List:
        for criterion in criterion_List:
            for direction, map_Tensor in [("Positive", positive_FSI_Map_Tensor), ("Negative", negative_FSI_Map_Tensor)]:
                fsi_Dict[criterion, direction, talker] = np.stack([
                    tf_Session.run( #Conducting above tensor.
                        fetches= map_Tensor,
                        feed_dict = {
                            avg_Activation_Placeholder: avg_Activation_Dict[talker][unit_Index],
                            criterion_Placeholder: criterion
                            }
                        ) for unit_Index in range(hidden_Size)],
                    axis=1
                    )

    return fsi_Dict

def Map_Squeezing(map_Dict):
    Squeezed_Dict = {}
    selected_Index_Dict = {}
    for key, map in map_Dict.items():
        selected_Index_Dict[key] = [index for index, sum_SI in enumerate(np.sum(map, axis=0)) if sum_SI > 0]    #Checking there is a significant PSI/FSI value
        if len(selected_Index_Dict[key]) == 0:  #If all PSI value of single row(unit) is 0, removed.
            selected_Index_Dict[key].append(0)
        Squeezed_Dict[key] = map[:, selected_Index_Dict[key]]

    return Squeezed_Dict, selected_Index_Dict

def Export_Map(map_Type, map_Dict, label_Dict, save_Path, prefix="", only_All = True):  #Export the PSI/FSI to text.
    os.makedirs(save_Path, exist_ok= True)
    os.makedirs(save_Path + "/TXT", exist_ok= True)
        
    for criterion, direction, talker in map_Dict.keys():
        if only_All and not talker == "All":    #If user use only all talker version.
            continue
        map = map_Dict[criterion, direction, talker]    #Getting map       
        if map_Type.upper() == "PSI".upper():
            row_Label_List = [phoneme_Label_Dict[phoneme] for phoneme in phoneme_List]
            column_Label_List = ["Phoneme"]
        elif map_Type.upper() == "FSI".upper():
            row_Label_List = feature_List
            column_Label_List = ["Feature"]
        else:
            raise ValueError("Not supported map type")
        column_Label_List.extend([str(x) for x in label_Dict[criterion, direction, talker]])
        
        extract_List = ["\t".join(column_Label_List)]
        for row_Label, row in zip(row_Label_List, map):
            extract_List.append("\t".join([row_Label] + [str(x) for x in row]))
        
        with open(os.path.join(save_Path, "TXT", "{}{}.C_{:.2f}.D_{}.T_{}.txt".format(prefix, map_Type.upper(), criterion, direction, talker)), "w", encoding='UTF8') as f:
            f.write("\n".join(extract_List))


def Phoneme_Flow_Dict_Generate(activation_Dict_by_Single_Phone):
    avg_Activation_Dict = {}
    
    avg_Activation_Dict["All"] = np.stack(
        [np.nanmean(activation_Dict_by_Single_Phone[phoneme], axis=0) for phoneme in phoneme_List],
        axis = 1
        ) #[Unit, Phoneme, Time]        
    for talker in talker_List:
        avg_Activation_Dict[talker] = np.stack(
            [np.nanmean(activation_Dict_by_Single_Phone[phoneme, talker], axis=0) for phoneme in phoneme_List],
            axis = 1
            )   #[Unit, Phoneme, Time]
        
    return avg_Activation_Dict

def Feature_Flow_Dict_Generate(activation_Dict_by_Feature):    
    avg_Activation_Dict = {}

    avg_Activation_Dict["All"] = np.stack(
        [np.nanmean(activation_Dict_by_Feature[feature], axis=0) for feature in feature_List],
        axis = 1
        ) #[Unit, Feature, Time]
    for talker in talker_List:
        avg_Activation_Dict[talker] = np.stack(
            [np.nanmean(activation_Dict_by_Feature[feature, talker], axis=0) for feature in feature_List],
            axis = 1
            ) #[Unit, Feature, Time]
        
    return avg_Activation_Dict

def Export_Flow(flow_Type, flow_Dict, save_Path, prefix="", only_All = True):
    os.makedirs(save_Path, exist_ok= True)
    os.makedirs(save_Path + "/TXT", exist_ok= True) # Flow save directory is generated if there is no directory.
        
    for talker in flow_Dict.keys():
        if only_All and not talker == "All":    #If user use only all talker version.
            continue
        flow = flow_Dict[talker]
        if flow_Type == "Phoneme":
            row_Label_List = [phoneme_Label_Dict[phoneme] for phoneme in phoneme_List]
            column_Label_List = ["Phoneme"]
        elif flow_Type == "Feature":
            row_Label_List = feature_List
            column_Label_List = ["Feature"]
        else:
            raise ValueError("Not supported flow type")
        column_Label_List.extend([str(x) for x in range(flow.shape[2])])
        
        for unit_Index, unit_Flow in enumerate(flow):
            extract_List = ["\t".join(column_Label_List)]
            for row_Label, row in zip(row_Label_List, unit_Flow):
                extract_List.append("\t".join([row_Label] + [str(x) for x in row]))
        
            with open(os.path.join(save_Path, "TXT", "{}{}.U_{}.T_{}.txt".format(prefix, flow_Type.upper(), unit_Index, talker)), "w", encoding='UTF8') as f:
                f.write("\n".join(extract_List))



def Export_Mean_Activation(activation_Dict, save_Path, prefix="", only_All = True): #This function is not used now. This is a preliminary function for situations that require analysis of activation values.
    os.makedirs(save_Path, exist_ok= True)
    os.makedirs(save_Path + "/TXT", exist_ok= True) # Save directory is generated if there is no directory.

    mean_Activation_Dict = {}
    for key, value in activation_Dict.items():
        if type(key) == str:
            key = (key, 'ALL')
        mean_Activation_Dict[key] = np.nanmean(value, axis=(0,2))  # [Unit, Phoneme], The activation is averaged by phoneme/feature and time steps, so the flow information disappear.

    hidden_Size_List = [x.shape[0] for x in mean_Activation_Dict.values()]
    hidden_Size = hidden_Size_List[0]
    if not all(hidden_Size == x for x in hidden_Size_List):
        assert False

    label_List = sorted(set([label for label, _ in mean_Activation_Dict.keys()]))   #Phoneme or feature list

    for talker in ['ALL'] + (talker_List if not only_All else []):
        extract_List = ["\t".join(["Label"] + [str(x) for x in range(hidden_Size)])]

        for label in label_List:
            extract_List.append('\t'.join([label] + [str(x) for x in mean_Activation_Dict[label, talker]]))
        
        with open(os.path.join(save_Path, "TXT", "{}.T_{}.txt".format(prefix, talker)), "w", encoding='UTF8') as f:
            f.write("\n".join(extract_List))
                
if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-d", "--extract_dir", required=True)
    argParser.add_argument("-e", "--epoch", required=True)
    argParser.add_argument("-v", "--voice_dir", required=True)
    argument_Dict = vars(argParser.parse_args())
    
    extract_Dir = argument_Dict["extract_dir"]
    selected_Epoch = int(argument_Dict["epoch"])
    
    new_Model = Model(  # Generating model
        start_Epoch=selected_Epoch,
        excluded_Talker = None,        
        extract_Dir=extract_Dir
        )
    new_Model.Restore(warning_Ignore = True)    #Loading model
    
    activation_Dict = {}
    map_Dict = {}
    label_Dict = {}
    cluster_Dict = {}
    sort_Index_List_Dict = {}

    criterion_List = [np.round(x, 2) for x in np.arange(0, 0.51, 0.01)] #[0.0, 0.15, 0.2, 0.44]

    #Activation dict
    activation_Dict["Phoneme"], activation_Dict["Feature"] = Activation_Dict_Generate(
        model = new_Model,
        voice_Path= argument_Dict["voice_dir"],
        )

    #Flow
    print('Flow exporting...')
    flow_Dict = {}
    flow_Dict["Phoneme"] = Phoneme_Flow_Dict_Generate(activation_Dict["Phoneme"])
    flow_Dict["Feature"] = Feature_Flow_Dict_Generate(activation_Dict["Feature"])

    for flow_Type in ["Phoneme", "Feature"]:
        Export_Flow(    #Extract flow text
            flow_Type= flow_Type,
            flow_Dict= flow_Dict[flow_Type],
            save_Path= extract_Dir + "/Hidden_Analysis/E.{}/Flow.{}".format(selected_Epoch, flow_Type),
            prefix= ''
            )

    #PSI, FSI
    #Process: Activation calculation -> PSI/FSI map generating -> Squeezing -> Exporting.
    print('PSI and FSI exporting...')    
        
    map_Dict["PSI", "Normal"] = PSI_Dict_Generate(model_Parameters.Hidden_Size, activation_Dict["Phoneme"], criterion_List = criterion_List)
    map_Dict["FSI", "Normal"] = FSI_Dict_Generate(model_Parameters.Hidden_Size, activation_Dict["Feature"], criterion_List = criterion_List)
    label_Dict["PSI", "Normal"] = {key: list(range(model_Parameters.Hidden_Size)) for key in map_Dict["PSI", "Normal"].keys()}
    label_Dict["FSI", "Normal"] = {key: list(range(model_Parameters.Hidden_Size)) for key in map_Dict["FSI", "Normal"].keys()}
                    
    for map_Type in ["PSI", "FSI"]:
        map_Dict[map_Type, "Squeezed"], label_Dict[map_Type, "Squeezed"] = Map_Squeezing(map_Dict[map_Type, "Normal"])
                    
    for map_Type in ["PSI", "FSI"]:
        for squeezing in ["Normal", "Squeezed"]:
            Export_Map(
                map_Type= map_Type,
                map_Dict= map_Dict[map_Type, squeezing],
                label_Dict= label_Dict[map_Type, squeezing],
                save_Path = extract_Dir + "/Hidden_Analysis/E.{}/Map.{}".format(selected_Epoch, map_Type),
                prefix= "{}.".format(squeezing),
                only_All= True
                )
