import numpy as np;
import tensorflow as tf;
import _pickle as pickle
import time, os, sys, zipfile, shutil, argparse;
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt;
from matplotlib.backends.backend_pdf import PdfPages
from EARS import EARShot_Model;
from Audio import *;
from Hyper_Parameters import pattern_Parameters, model_Parameters;
from Customized_Functions import Correlation2D, Batch_Correlation2D, Cosine_Similarity2D, Batch_Cosine_Similarity2D, MDS, Z_Score, Wilcoxon_Rank_Sum_Test2D, Mean_Squared_Error2D, Euclidean_Distance2D;

def Test_Pattern_Dict_Generate_from_TIMIT(timit_Path):
    if pattern_Parameters.Pattern_Use_Bit == 16:
        pattern_Bit_Type = np.float16;
    elif pattern_Parameters.Pattern_Use_Bit == 32:
        pattern_Bit_Type = np.float32;
    else:
        assert False;

    def Index_Generate(point, sample_Rate, frame_Length, frame_Shift):
        millisecond = point / sample_Rate * 1000
        return int(np.clip((millisecond - frame_Length + frame_Shift) / frame_Shift, a_max= np.inf, a_min= 0))

    timit_Pattern_Dict = {}
        
    for root, dirs, files in os.walk(timit_Path):
        root = root.upper();
        for file in files:            
            file = file.upper();
            if os.path.splitext(file)[1].upper() != ".WAV".upper():
                continue;

            talker_Code = root.replace("\\", "/").split("/")[-1];
            sentence_Code = os.path.splitext(file)[0].upper();
            
            voice_File_Path = os.path.join(root, file).replace("\\", "/");            
            phoneme_Alignment_File_Path = voice_File_Path.replace(os.path.splitext(voice_File_Path)[1].upper(), ".PHN").upper();
            word_Alignment_File_Path = voice_File_Path.replace(os.path.splitext(voice_File_Path)[1].upper(), ".WRD").upper();
            file_Sample_Rate = librosa.core.load(voice_File_Path)[1];
            
            print("Pattern generate: {}    {}".format(talker_Code, sentence_Code));

            if pattern_Parameters.Acoutsic_Mode.upper() == "Spectrogram".upper():                
                sig = librosa.core.load(voice_File_Path, sr = pattern_Parameters.Spectrogram.Sample_Rate)[0];
                acoustic_Pattern = normalize(np.transpose(spectrogram(sig)), min_level_db=-120).astype(pattern_Bit_Type);
                
                phoneme_Alignment_List = [];
                with open(phoneme_Alignment_File_Path, "r") as f:
                    alignment_Data = f.readlines();
                for alignment in alignment_Data:
                    start_Point, end_Point, word = alignment.upper().strip().split(" ");
                    start_Point = Index_Generate(
                        float(start_Point),
                        file_Sample_Rate,
                        pattern_Parameters.Spectrogram.Frame_Length,
                        pattern_Parameters.Spectrogram.Frame_Shift
                        )
                    end_Point = Index_Generate(
                        float(end_Point),
                        file_Sample_Rate,
                        pattern_Parameters.Spectrogram.Frame_Length,
                        pattern_Parameters.Spectrogram.Frame_Shift
                        )
                    phoneme_Alignment_List.append((start_Point, end_Point, word))
                
                word_Alignment_List = [];
                with open(word_Alignment_File_Path, "r") as f:
                    alignment_Data = f.readlines();
                for alignment in alignment_Data:
                    start_Point, end_Point, word = alignment.upper().strip().split(" ");
                    start_Point = Index_Generate(
                        float(start_Point),
                        file_Sample_Rate,
                        pattern_Parameters.Spectrogram.Frame_Length,
                        pattern_Parameters.Spectrogram.Frame_Shift
                        )
                    end_Point = Index_Generate(
                        float(end_Point),
                        file_Sample_Rate,
                        pattern_Parameters.Spectrogram.Frame_Length,
                        pattern_Parameters.Spectrogram.Frame_Shift
                        )
                    word_Alignment_List.append((start_Point, end_Point, word))
            else:
                assert False;
               
            if (talker_Code, sentence_Code) in timit_Pattern_Dict.keys():
                raise Exception("{}    {}".format(talker_Code, sentence_Code));

            timit_Pattern_Dict[talker_Code, sentence_Code] = {
                "Acoustic": acoustic_Pattern,
                "Phoneme_Alignment": phoneme_Alignment_List,
                "Word_Alignment": word_Alignment_List
                }

    with open("TIMIT_Pattern_Dict.pickle", "wb") as f:
        pickle.dump(timit_Pattern_Dict, f, protocol=0);

def Load_Metadata(metadata_File):
    print("Loading metadata...")
    metadata_Dict = {};

    with open(metadata_File, "rb") as f:
        load_Dict = pickle.load(f);
            
        metadata_Dict["Semantic_Size"] = load_Dict["Semantic_Size"];
        metadata_Dict["Pronunciation_Dict"] = load_Dict["Pronunciation_Dict"];

        metadata_Dict["Index_Dict"] = load_Dict["Test_Pattern_Dict"]["Index_Dict"]; #Key: (word, talker)
            
        metadata_Dict["Trained_Pattern_List"] = load_Dict["Trained_Pattern_List"];
        metadata_Dict["Excluded_Pattern_List"] = load_Dict["Excluded_Pattern_List"];
        metadata_Dict["Excluded_Talker"] = load_Dict["Excluded_Talker"];

        metadata_Dict["Word_Index_Dict"] = {};  #Semantic index(When you 1,000 words, the size of this dict becomes 1,000)
        metadata_Dict["Index_Word_Dict"] = {};
        target_List = [];
        for index, word in enumerate(list(set(word for word, _  in metadata_Dict["Trained_Pattern_List"]))):
            metadata_Dict["Word_Index_Dict"][word] = index;
            metadata_Dict["Index_Word_Dict"][index] = word;
            target_List.append(load_Dict["Target_Dict"][word]);
        metadata_Dict["Target_Array"] = np.vstack(target_List); #[Word, Semantic_Size]

    return metadata_Dict;


class Sentence_Analyzer:
    def __init__(
        self,
        folder,
        epoch,
        absolute_Criterion= None,
        relative_Criterion= None,
        time_Dependency_Criterion= None
        ):
        if pattern_Parameters.Pattern_Use_Bit == 16:
            self.np_Bit_Type = np.float16;
            self.tf_Bit_Type = tf.float16;
        elif pattern_Parameters.Pattern_Use_Bit == 32:
            self.np_Bit_Type = np.float32;
            self.tf_Bit_Type = tf.float32;
        else:
            assert False;

        self.absolute_Criterion= absolute_Criterion or 0.7
        self.relative_Criterion= relative_Criterion or 0.05,
        self.time_Dependency_Criterion= time_Dependency_Criterion or (5, 0.05)

                    
        self.ears_Model = EARShot_Model(
            excluded_Talker= None,
            start_Epoch= epoch,    #For restore            
            extract_Dir= folder
            );
        self.ears_Model.Restore(warning_Ignore = True);

        self.metadata_Dict = Load_Metadata(os.path.join(folder, "Result", "Metadata.pickle").replace("\\", "/"));
        
        self.tf_Session = self.ears_Model.tf_Session;        
        self.Tensor_Generate();

    def Tensor_Generate(self):
        result_Tensor = tf.placeholder(self.tf_Bit_Type, shape=[None, self.metadata_Dict["Semantic_Size"]]);  #[Cycle, Semantic_Size]
        target_Tensor = tf.constant(self.metadata_Dict["Target_Array"].astype(self.np_Bit_Type));  #[Pattern, Semantic_Size]
        tiled_Result_Tensor = tf.tile(tf.expand_dims(result_Tensor, [0]), multiples = [tf.shape(target_Tensor)[0], 1, 1]);   #[Pattern, Cycle, Semantic_Size]
        tiled_Target_Tensor = tf.tile(tf.expand_dims(target_Tensor, [1]), multiples = [1, tf.shape(result_Tensor)[0], 1]);   #[Pattern, Cycle, Semantic_Size]
        cosine_Similarity = tf.reduce_sum(tiled_Target_Tensor * tiled_Result_Tensor, axis = 2) / (tf.sqrt(tf.reduce_sum(tf.pow(tiled_Target_Tensor, 2), axis = 2)) * tf.sqrt(tf.reduce_sum(tf.pow(tiled_Result_Tensor, 2), axis = 2)))  #[Pattern, Cycle]

        self.semantic_Placeholder = result_Tensor
        self.cs_Tensor = cosine_Similarity;
    
    def Semantic_Activation_Generate(voice_File_Path_List):
        self.pattern_Index_Dict = {file_Path: index for index, file_Path in voice_File_Path_List};
        self.index_Pattern_Dict = {index: file_Path for index, file_Path in voice_File_Path_List};
        
        self.tf_Session.run(self.test_Mode_Turn_On_Tensor_List) #Backup the hidden state
        activation_List = [];
        for batch_Index in range(0, len(voice_File_Path_List), model_Parameters.Batch_Size):        
            _, semantic_Activation = ears_Model.tf_Session.run(
                fetches = ears_Model.test_Tensor_List,
                feed_dict = ears_Model.pattern_Feeder.Get_Test_Pattern_from_Voice(
                    voice_File_Path_List=voice_File_Path_List[batch_Index:batch_Index + model_Parameters.Batch_Size]
                    )
                )    #[Mini_Batch, Hidden, Time]
            activation_List.append(activation)
        self.tf_Session.run(self.test_Mode_Turn_Off_Tensor_List)     #Restore the hidden state
            
        activation_Array = np.zeros(shape=(
           np.sum([activation.shape[0]  for activation in activation_List]),
           np.max([activation.shape[1]  for activation in activation_List]),
           self.ears_Model.pattern_Feeder.semantic_Size
           ))

        current_Index = 0;
        for activation in activation_List:
            activation_Array[current_Index:current_Index+activation.shape[0], :activation.shape[1], :activation.shape[2]] = activation;
            current_Index += activation.shape[0]

        self.semantic_Activation = activation_Array;

    def Semantic_Activation_Generate_from_TIMIT(self, timit_Dict_Path):
        print("Generating semantic activation from TIMIT...")

        with open(timit_Dict_Path, "rb") as f:
            timit_Dict = pickle.load(f);

        self.pattern_Index_Dict = {};
        self.index_Pattern_Dict = {};
        self.alignment_Dict = {};
        acoustic_List = [];
        for index, ((talker_Code, sentence_Code), timit_Pattern_Dict) in enumerate(timit_Dict.items()):
            self.pattern_Index_Dict[talker_Code, sentence_Code] = index;
            self.index_Pattern_Dict[index] = (talker_Code, sentence_Code);
            self.alignment_Dict[index] = timit_Pattern_Dict["Word_Alignment"];
            acoustic_List.append(timit_Pattern_Dict["Acoustic"])

        def Acoustic_Array_Generate(batch_Start_Index, batch_End_Index):
            batch_Acoustic_List = acoustic_List[batch_Start_Index:batch_End_Index];
            max_Cycle = max([acoustic_Pattern.shape[0] for acoustic_Pattern in batch_Acoustic_List]);
            acoustic_Array = np.zeros(shape=(
                len(batch_Acoustic_List),
                max_Cycle,
                self.ears_Model.pattern_Feeder.acoustic_Size
                ))
            cycle_Array = np.ones((len(batch_Acoustic_List))).astype(np.int32) * max_Cycle;

            for index, acoustic_Pattern in enumerate(batch_Acoustic_List):
                acoustic_Array[index, :acoustic_Pattern.shape[0]] = acoustic_Pattern

            return {
                self.ears_Model.pattern_Feeder.placeholder_Dict["Acoustic"]: acoustic_Array,
                self.ears_Model.pattern_Feeder.placeholder_Dict["Length"]: cycle_Array
                }

        self.tf_Session.run(self.ears_Model.test_Mode_Turn_On_Tensor_List) #Backup the hidden state
        activation_List = [];
        for batch_Index in range(0, len(acoustic_List), model_Parameters.Batch_Size):        
            _, semantic_Activation = self.ears_Model.tf_Session.run(
                fetches = self.ears_Model.test_Tensor_List,
                feed_dict = Acoustic_Array_Generate(batch_Index, batch_Index + model_Parameters.Batch_Size)
                )    #[Mini_Batch, Hidden, Time]
            activation_List.append(semantic_Activation)
        self.tf_Session.run(self.ears_Model.test_Mode_Turn_Off_Tensor_List)     #Restore the hidden state
            
        activation_Array = np.zeros(shape=(
           np.sum([activation.shape[0]  for activation in activation_List]),
           np.max([activation.shape[1]  for activation in activation_List]),
           self.ears_Model.pattern_Feeder.semantic_Size
           ))
        
        current_Index = 0;
        for activation in activation_List:
            activation_Array[current_Index:current_Index+activation.shape[0], :activation.shape[1], :activation.shape[2]] = activation;
            current_Index += activation.shape[0]

        self.semantic_Activation = activation_Array;
          
    def TIMIT_Word_Checker(self, pattern_Index):
        for _, _, word in self.alignment_Dict[pattern_Index]:
            if not word in self.metadata_Dict["Word_Index_Dict"].keys():
                return False;

        return True;

    def Transform_CS(self, pattern_Index, cycle_Batch_Size = 400):        
        cs_Array_List = [];
        for batch_Index in range(0, self.semantic_Activation.shape[1], cycle_Batch_Size):
            cs_Array_List.append(self.tf_Session.run(
                self.cs_Tensor,
                feed_dict={self.semantic_Placeholder: self.semantic_Activation[batch_Index, batch_Index:batch_Index+cycle_Batch_Size]}
                ))

        return np.hstack(cs_Array_List);
    
    def Extract_Sentence_CS_Graph(self, pattern_Index, additional_Display_Word_List = []):
        cs_Array = self.Transform_CS(pattern_Index);
        alignment_List = self.alignment_Dict[pattern_Index];

        max_Cycle = cs_Array.shape[1];

        fig = plt.figure(figsize=(21, 9));

        #The words in sentence
        for word in list(set([sentence_Word for _, _, sentence_Word in alignment_List])):
            plt.plot(
                list(range(max_Cycle)),
                cs_Array[self.metadata_Dict["Word_Index_Dict"][word]],
                label = word,
                linewidth = 3.0
                )
        #Additional words
        for word in additional_Display_Word_List:
            if word in [sentence_Word for _, _, sentence_Word in alignment_List]:
                continue;
            plt.plot(
                list(range(max_Cycle)),
                cs_Array[self.metadata_Dict["Word_Index_Dict"][word]],
                label = word,
                linewidth = 3.0
                )
        #Other max
        total_Other_Max_Array = np.zeros((max_Cycle)) * np.nan;
        for start_Point, end_Point, word in alignment_List:
            other_Max_Array = np.max(np.delete(cs_Array, self.metadata_Dict["Word_Index_Dict"][word], 0), axis=0);            
            total_Other_Max_Array[start_Point:end_Point] = other_Max_Array[start_Point:end_Point];
        plt.plot(
            list(range(max_Cycle)),
            total_Other_Max_Array,
            label = "Other_Max",
            linewidth = 3.0
            )

        plt.gca().set_title(
            "Sentence: {}    Pattern_Code: {}".format(
                " ".join([sentence_Word for _, _, sentence_Word in alignment_List]),
                self.index_Pattern_Dict[pattern_Index]
                ),
            fontsize = 18
            )
        plt.gca().set_xlabel('Time (ms)', fontsize = 24);        
        plt.gca().set_ylabel('Cosine Similarity', fontsize = 24);
        plt.gca().set_xlim([0, max([end_Point for _, end_Point, _ in alignment_List])]);
        plt.gca().set_ylim([0, 1]);
        
        xtick_Dict = {};
        for start_Point, end_Point, word in alignment_List:
            if not start_Point in xtick_Dict.keys():
                xtick_Dict[start_Point] = "";
            if not end_Point in xtick_Dict.keys():
                xtick_Dict[end_Point] = "";
            xtick_Dict[start_Point] += "{}→\n".format(word);
            xtick_Dict[end_Point] += "←{}\n".format(word);

        xtick_List, xticklabel_List = list(zip(*xtick_Dict.items()));
        plt.gca().set_xticks(xtick_List);
        plt.gca().set_xticklabels(xticklabel_List);
        
        plt.legend(loc=1, ncol=1, fontsize=12);
        
        if not os.path.exists(os.path.join(self.ears_Model.extract_Dir, "Sentence").replace("\\", "/")):
            os.makedirs(os.path.join(self.ears_Model.extract_Dir, "Sentence").replace("\\", "/"));

        plt.savefig(os.path.join(self.ears_Model.extract_Dir, "Sentence", "{:06d}.png".format(pattern_Index)).replace("\\", "/"), bbox_inches='tight');
        plt.close(fig);
        
    def Extract_Sentence_CS_Accuracy(self, pattern_Index):
        cs_Array = self.Transform_CS(pattern_Index);
        alignment_List = self.alignment_Dict[pattern_Index];

        acc_Dict = {
            "Absolute": True,
            "Relative": True,
            "Time_dependent": True
            };
        for start_Point, end_Point, word in alignment_List:
            target_Array = cs_Array[self.metadata_Dict["Word_Index_Dict"][word]][start_Point:end_Point];
            other_Max_Array = np.max(np.delete(cs_Array, self.metadata_Dict["Word_Index_Dict"][word], 0), axis=0)[start_Point:end_Point];

            #Absolute
            if (other_Max_Array > self.absolute_Criterion).any() or (target_Array < self.absolute_Criterion).all():
                acc_Dict["Absolute"] = False;

            #Relative
            if not (target_Array > (other_Max_Array + self.relative_Criterion)).any():
                acc_Dict["Relative"] = False;

            #Time dependent
            time_Dependency_Criterion_Check = target_Array > (other_Max_Array + self.time_Dependency_Criterion[1]);
            time_Dependency_Sustain_Check = target_Array > other_Max_Array;
            time_Dependency_Decision = False;
            for cycle in range(target_Array.shape[0] - self.time_Dependency_Criterion[0]):
                if all(np.hstack([
                    time_Dependency_Criterion_Check[cycle:cycle+self.time_Dependency_Criterion[0]],
                    time_Dependency_Sustain_Check[cycle+self.time_Dependency_Criterion[0]:],
                    ])):
                    time_Dependency_Decision = True;
                    break;
            if not time_Dependency_Decision:
                acc_Dict["Time_dependent"] = False;
        
        if not os.path.exists(os.path.join(self.ears_Model.extract_Dir, "Sentence").replace("\\", "/")):
            os.makedirs(os.path.join(self.ears_Model.extract_Dir, "Sentence").replace("\\", "/"));

        accuracy_File_Path = os.path.join(self.ears_Model.extract_Dir, "Sentence", "Accuracy.txt").replace("\\", "/");

        if not os.path.exists(accuracy_File_Path):
            column_Title_List = [
                "Sentence",
                "Pattern_Code",
                "Absolute",
                "Relative",
                "Time_Dependent"
                ]
            with open(accuracy_File_Path, "w") as f:
                f.write("\t".join(column_Title_List) + "\n");

        line_List = [
            " ".join([sentence_Word for _, _, sentence_Word in alignment_List]),
            str(self.index_Pattern_Dict[pattern_Index]),
            str(int(acc_Dict["Absolute"])),
            str(int(acc_Dict["Relative"])),
            str(int(acc_Dict["Time_dependent"]))
            ]
        with open(accuracy_File_Path, "a") as f:
            f.write("\t".join(line_List) + "\n");


if __name__ == "__main__":
    Test_Pattern_Dict_Generate_from_TIMIT(timit_Path = "D:\Simulation_Raw_Data\EARS\TIMIT.FULL")
    
    argParser = argparse.ArgumentParser();
    argParser.add_argument("-f", "--folder", required=True);
    argParser.add_argument("-e", "--epoch", required=True);
    argParser.add_argument("-a", "--abs", required=False);
    argParser.add_argument("-r", "--rel", required=False);
    argParser.add_argument("-t", "--tim", required=False);
    argument_Dict = vars(argParser.parse_args());
    
    if argument_Dict["abs"] is not None:
        argument_Dict["abs"] = float(argument_Dict["abs"]);
    if argument_Dict["rel"] is not None:
        argument_Dict["rel"] = float(argument_Dict["rel"]);
    if argument_Dict["tim"] is not None:
        argument_Dict["tim"] = float(argument_Dict["tim"]);

    argument_Dict["epoch"] = int(argument_Dict["epoch"]);

    new_Sentence_Analyzer = Sentence_Analyzer(
        folder=argument_Dict["folder"],
        epoch=argument_Dict["epoch"],
        absolute_Criterion= argument_Dict["abs"] or 0.7,
        relative_Criterion= argument_Dict["rel"] or 0.05,
        time_Dependency_Criterion= argument_Dict["tim"] or (10, 0.05)
        )

    new_Sentence_Analyzer.Semantic_Activation_Generate_from_TIMIT("TIMIT_Pattern_Dict.pickle"); 

    for pattern_Index in range(new_Sentence_Analyzer.semantic_Activation.shape[0]):
        if new_Sentence_Analyzer.TIMIT_Word_Checker(pattern_Index):
            new_Sentence_Analyzer.Extract_Sentence_CS_Graph(pattern_Index);
            new_Sentence_Analyzer.Extract_Sentence_CS_Accuracy(pattern_Index);
        else:
            print("Index {} is skipped.".format(pattern_Index))
    