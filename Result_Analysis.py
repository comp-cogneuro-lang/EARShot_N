import numpy as np
import tensorflow as tf
import os, io, gc
import _pickle as pickle
import argparse
from Hyper_Parameters import pattern_Parameters, model_Parameters
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def Load_Metadata(metadata_File):   #Getting trained model's metadata.
    print("Loading metadata...")
    metadata_Dict = {}

    with open(metadata_File, "rb") as f:
        load_Dict = pickle.load(f)
            
        metadata_Dict["Semantic_Size"] = load_Dict["Semantic_Size"]
        metadata_Dict["Pronunciation_Dict"] = load_Dict["Pronunciation_Dict"]

        metadata_Dict["Index_Dict"] = load_Dict["Test_Pattern_Dict"]["Index_Dict"] #Key: (word, talker)
        metadata_Dict["Word_Talker_Dict"] = {value: key for key, value in metadata_Dict["Index_Dict"].items()} #Key: index, Value: (word, talker)
                    
        metadata_Dict["Trained_Pattern_List"] = load_Dict["Trained_Pattern_List"]
        metadata_Dict["Excluded_Pattern_List"] = load_Dict["Excluded_Pattern_List"]
        metadata_Dict["Excluded_Talker"] = load_Dict["Excluded_Talker"]

        metadata_Dict["Word_Index_Dict"] = {}  #Semantic index(When you 1,000 words, the size of this dict becomes 1,000)
        metadata_Dict["Index_Word_Dict"] = {}
        target_List = []
        for index, word in enumerate(list(set(word for word, _  in metadata_Dict["Trained_Pattern_List"]))):
            metadata_Dict["Word_Index_Dict"][word] = index
            metadata_Dict["Index_Word_Dict"][index] = word
            target_List.append(load_Dict["Target_Dict"][word])
        metadata_Dict["Target_Array"] = np.vstack(target_List) #[Word, Semantic_Size]
        
        metadata_Dict["Cycle_Array"] = load_Dict["Test_Pattern_Dict"]["Cycle_Pattern"] #[Pattern]
        metadata_Dict["Max_Cycle"] = int(np.max(metadata_Dict["Cycle_Array"]))

        metadata_Dict["Category_Dict"] = Category_Dict_Generate(metadata_Dict["Pronunciation_Dict"], metadata_Dict["Word_Index_Dict"])
        metadata_Dict["Pattern_Type_Dict"] = Pattern_Type_Dict_Generate(metadata_Dict["Trained_Pattern_List"], metadata_Dict["Excluded_Pattern_List"], metadata_Dict["Excluded_Talker"])
        metadata_Dict["Adjusted_Length_Dict"] = Adjusted_Length_Dict_Generate(metadata_Dict["Pronunciation_Dict"])

    return metadata_Dict

def Category_Dict_Generate(pronunciation_Dict, word_Index_Dict):    # Count the category of trained words.
    category_Dict = {}
    for target_Word in word_Index_Dict.keys():
        target_Pronunciation = pronunciation_Dict[target_Word]

        category_Dict[target_Word, "Target"] = []
        category_Dict[target_Word, "Cohort"] = []
        category_Dict[target_Word, "Rhyme"] = []
        category_Dict[target_Word, "DAS_Neighborhood"] = []
        category_Dict[target_Word, "Unrelated"] = []            

        for compare_Word, compare_Word_Index in word_Index_Dict.items():                
            compare_Pronunciation = pronunciation_Dict[compare_Word]

            unrelated = True

            if target_Word == compare_Word: #Target
                category_Dict[target_Word, "Target"].append(compare_Word_Index)
                unrelated = False
            if target_Pronunciation[0:2] == compare_Pronunciation[0:2] and target_Word != compare_Word: #Cohort
                category_Dict[target_Word, "Cohort"].append(compare_Word_Index)
                unrelated = False
            if target_Pronunciation[1:] == compare_Pronunciation[1:] and target_Pronunciation[0] != compare_Pronunciation[0] and target_Word != compare_Word:   #Rhyme
                category_Dict[target_Word, "Rhyme"].append(compare_Word_Index)
                unrelated = False
            if unrelated:
                category_Dict[target_Word, "Unrelated"].append(compare_Word_Index)  #Unrelated
            #For test
            if DAS_Neighborhood_Checker(target_Pronunciation, compare_Pronunciation):   #Neighborhood
                category_Dict[target_Word, "DAS_Neighborhood"].append(compare_Word_Index)

    return category_Dict

def DAS_Neighborhood_Checker(pronunciation1, pronunciation2):   #Delete, Addition, Substitution neighborhood checking
    #Same pronunciation
    if pronunciation1 == pronunciation2:
        return False

    #Exceed range
    elif abs(len(pronunciation1) - len(pronunciation2)) > 1:    #The length difference is bigger than 1, two pronunciations are not related.
        return False

    #Deletion
    elif len(pronunciation1) == len(pronunciation2) + 1:
        for index in range(len(pronunciation1)):
            deletion = pronunciation1[:index] + pronunciation1[index + 1:]
            if deletion == pronunciation2:
                return True

    #Addition
    elif len(pronunciation1) == len(pronunciation2) - 1:
        for index in range(len(pronunciation2)):
            deletion = pronunciation2[:index] + pronunciation2[index + 1:]
            if deletion == pronunciation1:
                return True

    #Substitution
    elif len(pronunciation1) == len(pronunciation2):
        for index in range(len(pronunciation1)):
            pronunciation1_Substitution = pronunciation1[:index] + pronunciation1[index + 1:]
            pronunciation2_Substitution = pronunciation2[:index] + pronunciation2[index + 1:]
            if pronunciation1_Substitution == pronunciation2_Substitution:
                return True

    return False

def Pattern_Type_Dict_Generate(trained_Pattern_List, excluded_Pattern_List, excluded_Talker):   #Generating the pattern type dictionary.
    pattern_Type_Dict = {}
    for (word, talker) in trained_Pattern_List:
        pattern_Type_Dict[word, talker] = "Trained"
    for (word, talker) in excluded_Pattern_List:    #There are three types of excluded patterns: test only, talker and pattern
        if talker == excluded_Talker:
            pattern_Type_Dict[word, talker] = "Talker_Excluded"
        elif not model_Parameters.Test_Only_Identifier_List is None and talker in model_Parameters.Test_Only_Identifier_List:
            pattern_Type_Dict[word, talker] = "Test_Only"  #In simulation, this is same to talker excluded.
        else:
            pattern_Type_Dict[word, talker] = "Pattern_Excluded"

    return pattern_Type_Dict

def Adjusted_Length_Dict_Generate(pronunciation_Dict): #For uniqueness point.
    adjusted_Length_Dict = {}

    for word, pronunciation in pronunciation_Dict.items():
        for cut_Length in range(1, len(pronunciation) + 1):
            cut_Pronunciation = pronunciation[:cut_Length]
            cut_Comparer_List = [comparer[:cut_Length] for comparer in pronunciation_Dict.values() if pronunciation != comparer]
            if not cut_Pronunciation in cut_Comparer_List:  #When you see a part of target phoneme string, if there is no other competitor.
                adjusted_Length_Dict[word] = cut_Length - len(pronunciation) - 1
                break
        if not word in adjusted_Length_Dict.keys():
            adjusted_Length_Dict[word] = 0

    return adjusted_Length_Dict

class Result_Analyzer:
    def __init__(
        self,
        metadata_Dict,
        cycle_Cut=False,
        absolute_Criterion=None,
        relative_Criterion=None,
        time_Dependency_Criterion=None
        ):
        self.tf_Session = tf.Session()  # analyzer need high performance to calculate all cosine simiarities between distributed output vector and all words' target patterns.

        self.metadata_Dict = metadata_Dict
        self.cycle_Cut = cycle_Cut  #If the effect of zero padding is ignored, it is true
        self.absolute_Criterion = absolute_Criterion or 0.7
        self.relative_Criterion = relative_Criterion or 0.05
        self.time_Dependency_Criterion = time_Dependency_Criterion or (10, 0.05)
        
        self.Tensor_Generate()
        
    
    def Tensor_Generate(self):
        #I think this line do not need anymore because TF 1.x does not support 16bit well.
        if pattern_Parameters.Pattern_Use_Bit == 16:
            self.bit_Type = tf.float16
        elif pattern_Parameters.Pattern_Use_Bit == 32:
            self.bit_Type = tf.float32
        else:
            assert False

        result_Tensor = tf.placeholder(self.bit_Type, shape=[None, self.metadata_Dict["Semantic_Size"]])  #[Cycle, Semantic_Size], placeholder is variable space. Output vector is inputted by this placeholder
        target_Tensor = tf.constant(self.metadata_Dict["Target_Array"], dtype=self.bit_Type)  #[Pattern, Semantic_Size]. To compare
        tiled_Result_Tensor = tf.tile(tf.expand_dims(result_Tensor, [0]), multiples = [tf.shape(target_Tensor)[0], 1, 1])   #[Pattern, Cycle, Semantic_Size], increase dimension and tiled for 2D comparing.
        tiled_Target_Tensor = tf.tile(tf.expand_dims(target_Tensor, [1]), multiples = [1, tf.shape(result_Tensor)[0], 1])   #[Pattern, Cycle, Semantic_Size], increase dimension and tiled for 2D comparing.
        cosine_Similarity = tf.reduce_sum(tiled_Target_Tensor * tiled_Result_Tensor, axis = 2) / (tf.sqrt(tf.reduce_sum(tf.pow(tiled_Target_Tensor, 2), axis = 2)) * tf.sqrt(tf.reduce_sum(tf.pow(tiled_Result_Tensor, 2), axis = 2)))  #[Pattern, Cycle]

        self.semantic_Placeholder = result_Tensor
        self.cs_Tensor = cosine_Similarity

    def Loading_Results(self, result_File): #Pickled result is loaded.
        print("Loading: {}".format(result_File))
        with open(result_File, "rb") as f:
            result_Dict = pickle.load(f)
        
        self.result_Epoch = result_Dict["Epoch"]
        self.result_Start_Index = result_Dict["Start_Index"]
        self.result_Array = result_Dict["Result"]

        self.extract_Dir_Name = os.path.dirname(result_File)


    def Data_Generate_by_CS(self, target_Index, cycle_Batch_Size = 200):    #Target pattern is distributed like SRV or PGD.
        cs_Array_List = []
        for batch_Index in range(0, self.result_Array.shape[1], cycle_Batch_Size):
            cs_Array_List.append(self.tf_Session.run(
                self.cs_Tensor,
                feed_dict={self.semantic_Placeholder: self.result_Array[target_Index - self.result_Start_Index, batch_Index:batch_Index+cycle_Batch_Size]}
                ))  #Calculating the cosine similarity between output vector and target array.
        cs_Array = np.hstack(cs_Array_List) #Stack the time steps of CS Array

        if self.cycle_Cut:  #If zero padding is ignored.
            cs_Array[:, int(self.metadata_Dict["Cycle_Array"][target_Index]):] = cs_Array[:, [int(self.metadata_Dict["Cycle_Array"][target_Index]) - 1]]

        return cs_Array

    def Data_Generate_by_Activation(self, target_Index):    #Target pattern is one-hot.
        activation_Array = np.transpose(self.result_Array[target_Index - self.result_Start_Index])

        if self.cycle_Cut:            
            activation_Array[:, int(self.metadata_Dict["Cycle_Array"][target_Index]):] = activation_Array[:, [int(self.metadata_Dict["Cycle_Array"][target_Index]) - 1]]

        return activation_Array

    def RT_Generate(self, word, talker, data):
        rt_Dict = {
            ("Onset", "Absolute"): np.nan,
            ("Onset", "Relative"): np.nan,
            ("Onset", "Time_Dependent"): np.nan
            }

        target_Index = self.metadata_Dict["Word_Index_Dict"][word]
        target_Array = data[target_Index]
        other_Max_Array = np.max(np.delete(data, target_Index, 0), axis=0)  #Target is removed, and using the max value of each time step.
            
        #Absolute threshold RT
        if not (other_Max_Array > self.absolute_Criterion).any():
            absolute_Check_Array = target_Array > self.absolute_Criterion
            for cycle in range(self.metadata_Dict["Max_Cycle"]):
                if absolute_Check_Array[cycle]:
                    rt_Dict["Onset", "Absolute"] = cycle
                    break

        #Relative threshold RT
        relative_Check_Array = target_Array > (other_Max_Array + self.relative_Criterion)            
        for cycle in range(self.metadata_Dict["Max_Cycle"]):
            if relative_Check_Array[cycle]:
                rt_Dict["Onset", "Relative"] = cycle
                break

        #Time dependent RT
        time_Dependency_Check_Array_with_Criterion = target_Array > other_Max_Array + self.time_Dependency_Criterion[1]
        time_Dependency_Check_Array_Sustainment = target_Array > other_Max_Array
        for cycle in range(self.metadata_Dict["Max_Cycle"] - self.time_Dependency_Criterion[0]):
            if all(np.hstack([
                time_Dependency_Check_Array_with_Criterion[cycle:cycle + self.time_Dependency_Criterion[0]],
                time_Dependency_Check_Array_Sustainment[cycle + self.time_Dependency_Criterion[0]:]
                ])):
                rt_Dict["Onset", "Time_Dependent"] = cycle
                break

        #Offset_RT = Onset_RT - length
        if not np.isnan(rt_Dict["Onset", "Absolute"]):
            rt_Dict["Offset", "Absolute"] = rt_Dict["Onset", "Absolute"] - self.metadata_Dict["Cycle_Array"][self.metadata_Dict["Index_Dict"][word, talker]]
        else:
            rt_Dict["Offset", "Absolute"] = rt_Dict["Onset", "Absolute"]    #np.nan
        if not np.isnan(rt_Dict["Onset", "Relative"]):
            rt_Dict["Offset", "Relative"] = rt_Dict["Onset", "Relative"] - self.metadata_Dict["Cycle_Array"][self.metadata_Dict["Index_Dict"][word, talker]]
        else:
            rt_Dict["Offset", "Relative"] = rt_Dict["Onset", "Relative"]    #np.nan
        if not np.isnan(rt_Dict["Onset", "Time_Dependent"]):
            rt_Dict["Offset", "Time_Dependent"] = rt_Dict["Onset", "Time_Dependent"] - self.metadata_Dict["Cycle_Array"][self.metadata_Dict["Index_Dict"][word, talker]]
        else:
            rt_Dict["Offset", "Time_Dependent"] = rt_Dict["Onset", "Time_Dependent"]    #np.nan

        return rt_Dict

    def Category_Flow_Generate(self, word, talker, data):   #For categorized flow
        category_Flow_Dict = {}
        
        for category in ["Target", "Cohort", "Rhyme", "Unrelated"]:
            if len(self.metadata_Dict["Category_Dict"][word, category]) > 0:
                category_Flow_Dict[category] = np.mean(data[self.metadata_Dict["Category_Dict"][word, category],:], axis=0) #Calculation mean of several same category flows.
            else:
                category_Flow_Dict[category] = np.zeros((data.shape[1])) * np.nan   # If there is no word which is belonged a specific category, nan value.
                
        category_Flow_Dict["All"] = np.mean(data, axis=0)
        category_Flow_Dict["Other_Max"] = np.max(np.delete(data, self.metadata_Dict["Word_Index_Dict"][word], 0), axis=0)   #Target is removed, and using the max value of each time step.

        return category_Flow_Dict


    def Result_Write(self, reaction_Time = True, categorized_Flow = True, top10_Flow = False, raw_Data = False):
        # Storing
        self.rt_Extract_List = []
        self.raw_Data_Extract_List = []
        self.categorized_Flow_Extract_List = []

        for target_Index in range(self.result_Start_Index, self.result_Start_Index + self.result_Array.shape[0]):
            word, talker = self.metadata_Dict["Word_Talker_Dict"][target_Index]

            if pattern_Parameters.Semantic_Mode in ["PGD", "SRV"]:
                data = self.Data_Generate_by_CS(target_Index)   #Getting CS array
            elif pattern_Parameters.Semantic_Mode == "One-hot":
                data = self.Data_Generate_by_Activation(target_Index)   #Getting activation array

            rt_Dict = self.RT_Generate(word, talker, data)  #Getting reaction time

            if reaction_Time:   
                self.Extract_RT(word, talker, rt_Dict)

            if categorized_Flow:    
                category_Flow_Dict = self.Category_Flow_Generate(word, talker, data)
                self.Extract_Categorized_Flow(word, talker, rt_Dict, category_Flow_Dict)

            if top10_Flow:  
                self.Extract_Top10_Flow(word, talker, data, rt_Dict)

            if raw_Data:
                self.Extract_Raw_Data(word, talker, data)

        if reaction_Time:   #Extract reaction time text
            rt_File_Path = os.path.join(self.extract_Dir_Name, "RT_Result.txt").replace("\\", "/")
            if not os.path.exists(rt_File_Path):
                column_Title_List = [
                    "Epoch",
                    "Word",
                    "Talker",
                    "Pattern_Type",
                    "Length",
                    "Adjusted_Length",
                    "Cohort",
                    "Rhyme",
                    "DAS_Neighborhood",            
                    "Onset_Absolute_RT",
                    "Onset_Relative_RT",
                    "Onset_Time_Dependent_RT",
                    "Offset_Absolute_RT",
                    "Offset_Relative_RT",
                    "Offset_Time_Dependent_RT"
                    ]
                self.rt_Extract_List = ["\t".join(column_Title_List)] + self.rt_Extract_List

            with open(rt_File_Path, "a") as f:
                f.write("\n".join(self.rt_Extract_List) + "\n")

        if categorized_Flow:    #Extract categorized flow text
            categorized_Flow_Path = os.path.join(self.extract_Dir_Name, "Categorized_Flow").replace("\\", "/")
            if not os.path.exists(categorized_Flow_Path):
                os.makedirs(categorized_Flow_Path)
            categorized_Flow_File_Path = os.path.join(categorized_Flow_Path, "Categorized_Flow.E_{}.txt".format(self.result_Epoch)).replace("\\", "/")
            if not os.path.exists(categorized_Flow_File_Path):
                column_Title_List = [
                    "Epoch",
                    "Word",
                    "Talker",
                    "Pattern_Type",
                    "Length",
                    "Adjusted_Length",
                    "Category",
                    "Category_Count",
                    "Accuracy"
                    ] + [str(x) for x in range(metadata_Dict["Max_Cycle"])]
                self.categorized_Flow_Extract_List = ["\t".join(column_Title_List)] + self.categorized_Flow_Extract_List

            with open(categorized_Flow_File_Path, "a") as f:
                f.write("\n".join(self.categorized_Flow_Extract_List) + "\n")

        if raw_Data:    #Extract raw data text
            raw_Data_File_Path = os.path.join(self.extract_Dir_Name, "Raw_Data.E_{}.txt".format(self.result_Epoch)).replace("\\", "/")
            if not os.path.exists(raw_Data_File_Path):
                column_Title_List = [
                    "Epoch",
                    "Target_Word",
                    "Talker",
                    "Pattern_Type",
                    "Pattern_Length",
                    "Compare_Word"
                    ] + [str(x) for x in range(metadata_Dict["Max_Cycle"])]
                self.raw_Data_Extract_List += ["\t".join(column_Title_List)] + self.raw_Data_Extract_List

            with open(raw_Data_File_Path, "a") as f:
                f.write("\n".join(self.raw_Data_Extract_List) + "\n")

    def Extract_RT(self, word, talker, rt_Dict):    # Generating rt text line
        line_List = [            
            str(self.result_Epoch),
            word,
            talker,
            self.metadata_Dict["Pattern_Type_Dict"][word, talker],
            str(len(self.metadata_Dict["Pronunciation_Dict"][word])),
            str(self.metadata_Dict["Adjusted_Length_Dict"][word]),
            str(len(self.metadata_Dict["Category_Dict"][word, "Cohort"])),
            str(len(self.metadata_Dict["Category_Dict"][word, "Rhyme"])),
            str(len(self.metadata_Dict["Category_Dict"][word, "DAS_Neighborhood"])),
            str(rt_Dict["Onset", "Absolute"]),
            str(rt_Dict["Onset", "Relative"]),
            str(rt_Dict["Onset", "Time_Dependent"]),
            str(rt_Dict["Offset", "Absolute"]),
            str(rt_Dict["Offset", "Relative"]),
            str(rt_Dict["Offset", "Time_Dependent"])
            ]
        self.rt_Extract_List.append("\t".join(line_List))

    def Extract_Raw_Data(self, word, talker, data): # Generating raw data text lines
        target_Word_Index = self.metadata_Dict["Word_Index_Dict"][word]
        data = np.round(data, 5)
        
        for compare_Word, compare_Word_Index in self.metadata_Dict["Word_Index_Dict"].items():
            line_List = [
                str(self.result_Epoch),
                word,
                talker,
                self.metadata_Dict["Pattern_Type_Dict"][word, talker],
                str(self.metadata_Dict["Cycle_Array"][target_Word_Index]),
                compare_Word
                ]
            line_List += [str(x) for x in data[compare_Word_Index, :]]
            self.raw_Data_Extract_List.append("\t".join(line_List))

    def Extract_Categorized_Flow(self, word, talker, rt_Dict, category_Flow_Dict):  #Generating categorized text lines
        for category in ["Target", "Cohort", "Rhyme", "Unrelated", "Other_Max"]:
            if category == "Other_Max":
                category_Count = np.nan
            else:
                category_Count = len(self.metadata_Dict["Category_Dict"][word, category])

            line_List = [            
                str(self.result_Epoch),
                word,
                talker,
                self.metadata_Dict["Pattern_Type_Dict"][word, talker],
                str(len(self.metadata_Dict["Pronunciation_Dict"][word])),
                str(self.metadata_Dict["Adjusted_Length_Dict"][word]),
                category,
                str(category_Count),
                str(not np.isnan(rt_Dict["Onset", "Time_Dependent"])).upper()
                ]
            line_List += [str(np.round(x, 5)) for x in category_Flow_Dict[category]]
            self.categorized_Flow_Extract_List.append("\t".join(line_List))

    def Extract_Top10_Flow(self, word, talker, data, rt_Dict): #Extract top10 flow text
        top10_Flow_Path = os.path.join(self.extract_Dir_Name, "TOP10", "E_{}".format(self.result_Epoch)).replace("\\", "/")

        if not os.path.exists(top10_Flow_Path):
            os.makedirs(top10_Flow_Path)

        if self.cycle_Cut:
            cycle_Cut = int(self.metadata_Dict["Cycle_Array"][self.metadata_Dict["Index_Dict"][word, talker]])
        else:
            cycle_Cut = self.metadata_Dict["Max_Cycle"]
                    
        top10_Flow_List = [
            "\t".join(["Epoch", str(self.result_Epoch)]),
            "\t".join(["Word", word]),
            "\t".join(["Talker", talker]),            
            "\t".join(["Absolute_RT", str(rt_Dict["Onset", "Absolute"])]),
            "\t".join(["Relative_RT", str(rt_Dict["Onset", "Relative"])]),
            "\t".join(["Time_Dependent_RT", str(rt_Dict["Onset", "Time_Dependent"])]),
            ""
            ]

        target_Index = self.metadata_Dict["Word_Index_Dict"][word]
        extract_Index_List = [target_Index]

        sorted_Indices = np.argsort(np.nanmax(data, axis=1))[-11:]            
        sorted_Indices = np.delete(sorted_Indices, np.where(sorted_Indices == target_Index))[-10:]
        extract_Index_List.extend(np.flip(sorted_Indices, axis=0))

        column_Title_List = [
            "Word",
            "Target",
            "Cohort",
            "Rhyme"
            ] + [str(x) for x in range(cycle_Cut)]
        top10_Flow_List.append("\t".join(column_Title_List))
        
        for index in extract_Index_List:
            line_List = [
                self.metadata_Dict["Index_Word_Dict"][index],
                str(index in self.metadata_Dict["Category_Dict"][word, "Target"]).upper(),
                str(index in self.metadata_Dict["Category_Dict"][word, "Cohort"]).upper(),
                str(index in self.metadata_Dict["Category_Dict"][word, "Rhyme"]).upper()
                ]
            line_List += [str(np.round(x, 5)) for x in data[index, :cycle_Cut]]
            top10_Flow_List.append("\t".join(line_List))

        with open(os.path.join(top10_Flow_Path, "W_{}.T_{}.E_{:06d}.txt").format(word, talker, self.result_Epoch).replace("\\", "/"), "w") as f:
            f.write("\n".join(top10_Flow_List))

if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-d", "--extract_dir", required=False)
    argParser.add_argument("-a", "--abs", required=False)
    argParser.add_argument("-r", "--rel", required=False)
    argParser.add_argument("-tw", "--tim_width", required=False)
    argParser.add_argument("-th", "--tim_height", required=False)
    argument_Dict = vars(argParser.parse_args())

    if argument_Dict["abs"] is not None:
        argument_Dict["abs"] = float(argument_Dict["abs"])
    if argument_Dict["rel"] is not None:
        argument_Dict["rel"] = float(argument_Dict["rel"])
    if argument_Dict["tim_height"] is not None:
        argument_Dict["tim_height"] = float(argument_Dict["tim_height"])
    if argument_Dict["tim_width"] is not None:
        argument_Dict["tim_width"] = float(argument_Dict["tim_width"])
        
    #Loading result metadata
    metadata_Dict = Load_Metadata(os.path.join(argument_Dict["extract_dir"], "Result", "Metadata.pickle").replace("\\", "/"))

    new_Result_Analyzer = Result_Analyzer(  #Generating result analyzer object
        metadata_Dict = metadata_Dict,
        cycle_Cut=True,
        absolute_Criterion= argument_Dict["abs"] or 0.7,
        relative_Criterion= argument_Dict["rel"] or 0.05,
        time_Dependency_Criterion = (argument_Dict["tim_width"] or 10, argument_Dict["tim_height"] or 0.05)
        )
    
    result_File_List = sorted([ #Result files sorting
       os.path.join(argument_Dict["extract_dir"], "Result", x).replace("\\", "/") for x in os.listdir(os.path.join(argument_Dict["extract_dir"], "Result").replace("\\", "/"))
       if x.endswith(".pickle") and x != 'Metadata.pickle'
       ])
    for result_File in result_File_List:    #Extract results
        new_Result_Analyzer.Loading_Results(result_File)
        new_Result_Analyzer.Result_Write(reaction_Time=True, categorized_Flow=True, top10_Flow=False, raw_Data=False)