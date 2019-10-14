import numpy as np
import tensorflow as tf
import _pickle as pickle
from threading import Thread
from collections import deque, Sequence
from random import shuffle
import time, librosa, os
from Audio import *
from Hyper_Parameters import pattern_Parameters, model_Parameters

class Pattern_Feeder:
    def __init__(
        self,
        start_Epoch,
        excluded_Talker = None,
        metadata_File = None
        ):
        self.start_Epoch = start_Epoch
        if isinstance(excluded_Talker, str):
            self.excluded_Talker = excluded_Talker.upper()
        elif isinstance(excluded_Talker, Sequence):
            self.excluded_Talker = [talker.upper() for talker in excluded_Talker]
        else:
            self.excluded_Talker = None
        
        self.Pattern_Metadata_Load()
        self.Placeholder_Generate()
        
        self.is_Finished = False
        self.is_Test_Pattern_Generated = False
        self.pattern_Queue = deque()

        if metadata_File is None:
            self.Training_Pattern_Path_Dict_Generate()
            test_Pattern_Generate_Thread = Thread(target=self.Test_Pattern_Generate)
            test_Pattern_Generate_Thread.daemon = True
            test_Pattern_Generate_Thread.start()            
        else:
            self.Load_Metadata(metadata_File)

        pattern_Generate_Thread = Thread(target=self.Pattern_Generate)
        pattern_Generate_Thread.daemon = True
        pattern_Generate_Thread.start()
                    
    def Placeholder_Generate(self):
        if pattern_Parameters.Pattern_Use_Bit == 16:
            float_Bit_Type = tf.float16
            int_Bit_Type = tf.int16
        elif pattern_Parameters.Pattern_Use_Bit == 32:
            float_Bit_Type = tf.float32
            int_Bit_Type = tf.int32
        else:
            assert False

        with tf.variable_scope('placeHolders') as scope:            
            self.placeholder_Dict = {
                "Is_Training": tf.placeholder(tf.bool, name='is_Training_Placeholder'),
                "Acoustic": tf.placeholder(float_Bit_Type, shape=(None, None, self.acoustic_Size), name = "acoustic_Placeholder"), #(batch, length, size)            
                "Semantic": tf.placeholder(float_Bit_Type, shape=(None, None, self.semantic_Size), name = "semantic_Placeholder"), #(batch, length, size) 'Length' is for compatibility with sentence patterns.
                "Length": tf.placeholder(tf.int32, shape=(None,), name = "length_Placeholder"),   #(batch)
                }

    def Pattern_Metadata_Load(self):
        with open (os.path.join(pattern_Parameters.Pattern_Path, pattern_Parameters.Pattern_Metadata_File_Name).replace("\\", "/"), "rb") as f:
            load_Dict = pickle.load(f)
        
        self.acoustic_Size = load_Dict["Acoustic_Size"]
        self.semantic_Size = load_Dict["Semantic_Size"]
        self.pronunciation_Dict = load_Dict["Pronunciation_Dict"]
        self.pattern_Path_Dict = load_Dict["Pattern_Path_Dict"]
        self.word_and_Identifier_Dict = load_Dict["Word_and_Identifier_Dict"]
        self.cycle_Dict = load_Dict["Cycle_Dict"]
        self.target_Dict = load_Dict["Target_Dict"]

    def Load_Metadata(self, metadata_File):
        if pattern_Parameters.Pattern_Use_Bit == 16:
            float_Bit_Type = np.float16
            int_Bit_Type = np.int16
        elif pattern_Parameters.Pattern_Use_Bit == 32:
            float_Bit_Type = np.float32
            int_Bit_Type = np.int32
        else:
            assert False

        with open (metadata_File, "rb") as f:
            metadata_Dict = pickle.load(f)

        self.training_Pattern_Path_Dict = {key:self.pattern_Path_Dict[key] for key in metadata_Dict["Trained_Pattern_List"]}
        self.excluded_Pattern_Path_Dict = {key:self.pattern_Path_Dict[key] for key in metadata_Dict["Excluded_Pattern_List"]}

        self.test_Pattern_Dict = metadata_Dict["Test_Pattern_Dict"]
        
        self.test_Pattern_Dict["Acoustic_Pattern"] = np.zeros((self.test_Pattern_Dict["Count"], self.test_Pattern_Dict["Max_Cycle"], self.acoustic_Size)).astype(float_Bit_Type)
        self.test_Pattern_Dict["Semantic_Pattern"] = np.zeros((self.test_Pattern_Dict["Count"], self.test_Pattern_Dict["Max_Cycle"], self.semantic_Size)).astype(float_Bit_Type)
        
        for (word, talker), index in metadata_Dict["Test_Pattern_Dict"]["Index_Dict"].items():
            pattern_Path = self.pattern_Path_Dict[word, talker]
            with open (os.path.join(pattern_Parameters.Pattern_Path, pattern_Path).replace("\\", "/"), "rb") as f:
                load_Dict = pickle.load(f)

            self.test_Pattern_Dict["Acoustic_Pattern"][index, :load_Dict["Acoustic"].shape[0]] = load_Dict["Acoustic"]
            self.test_Pattern_Dict["Semantic_Pattern"][index, :load_Dict["Semantic"].shape[0]] = load_Dict["Semantic"]

            if self.test_Pattern_Dict["Cycle_Pattern"][index] != load_Dict["Acoustic"].shape[0]:
                raise ValueError("Pattern inconsistency!")

        self.test_Pattern_Dict["Feed_Dict_List"] = []
        for start_Index in range(0, self.test_Pattern_Dict["Count"], model_Parameters.Batch_Size):
            test_Pattern = self.test_Pattern_Dict["Acoustic_Pattern"][start_Index:start_Index + model_Parameters.Batch_Size]
            test_Length = np.ones((test_Pattern.shape[0])) * self.test_Pattern_Dict["Max_Cycle"]
            new_Feed_Dict= {
                self.placeholder_Dict["Acoustic"]: test_Pattern,
                self.placeholder_Dict["Length"]: test_Length
                }
            self.test_Pattern_Dict["Feed_Dict_List"].append(new_Feed_Dict)

        self.is_Test_Pattern_Generated = True


    def Training_Pattern_Path_Dict_Generate(self):
        '''
        When 'model_Parameters.Exclusion_Mode' is 'P'(Pattern based), each talker's partial pattern will not be trained.
        When 'model_Parameters.Exclusion_Mode' is 'T'(Talker based), a talker's all pattern will not be trained.        
        When 'model_Parameters.Exclusion_Mode' is 'M'(Mix based), each talker's partial pattern will not be trained and a talker's all pattern will not be trained.
        When 'model_Parameters.Exclusion_Mode' is None, all pattern will be trained.

        The talkers who are in hyper parameter 'Test_Only_Identifier_List' are always excluded.
        '''

        self.training_Pattern_Path_Dict = {}
        self.excluded_Pattern_Path_Dict = {}
            
        #Test_Only_Identifier_List hyper parameter
        test_Only_Identifier_List = model_Parameters.Test_Only_Identifier_List or []
        self.excluded_Pattern_Path_Dict.update({
            (word, talker): pattern_Path
            for (word, talker), pattern_Path in self.pattern_Path_Dict.items()
            if talker in test_Only_Identifier_List
            })

        if model_Parameters.Exclusion_Mode is None:
            self.training_Pattern_Path_Dict = {
                (word, talker): pattern_Path
                for (word, talker), pattern_Path in self.pattern_Path_Dict.items()
                if not talker in test_Only_Identifier_List
                }
            return
        
        talker_List = list(set([talker for word, talker in self.pattern_Path_Dict.keys() if not talker in test_Only_Identifier_List]))
        shuffle(talker_List)
        word_List = list(set([word for word, talker in self.pattern_Path_Dict.keys()]))
        shuffle(word_List)
        
        if model_Parameters.Exclusion_Mode.upper() == 'P':            
            exclude_Size = len(word_List) // len(talker_List)
            for talker_Index, talker in enumerate(talker_List):
                for word in word_List[:talker_Index * exclude_Size] + word_List[(talker_Index + 1) * exclude_Size:]:
                    self.training_Pattern_Path_Dict[word, talker] = self.pattern_Path_Dict[word, talker]
                for word in word_List[talker_Index * exclude_Size:(talker_Index + 1) * exclude_Size]:
                    self.excluded_Pattern_Path_Dict[word, talker] = self.pattern_Path_Dict[word, talker]
            return

        #Select excluded talker
        if not self.excluded_Talker is None:
            if not self.excluded_Talker.upper() in talker_List:
                raise Exception("The specified talker is not in list.")                    
        else:
            self.excluded_Talker = talker_List[-1]
        talker_List.remove(self.excluded_Talker)
          
        if model_Parameters.Exclusion_Mode.upper() == 'T':
            for word in word_List:
                for talker in talker_List:
                    self.training_Pattern_Path_Dict[word, talker] = self.pattern_Path_Dict[word, talker]
                self.excluded_Pattern_Path_Dict[word, self.excluded_Talker] = self.pattern_Path_Dict[word, self.excluded_Talker]
            return

        if model_Parameters.Exclusion_Mode.upper() == 'M':
            exclude_Size = len(word_List) // len(talker_List)  #It is different from 'P' mode because of talker exclusion.
            for talker_Index, talker in enumerate(talker_List): #Pattern exclusion
                for word in word_List[:talker_Index * exclude_Size] + word_List[(talker_Index + 1) * exclude_Size:]:
                    self.training_Pattern_Path_Dict[word, talker] = self.pattern_Path_Dict[word, talker]
                for word in word_List[talker_Index * exclude_Size:(talker_Index + 1) * exclude_Size]:
                    self.excluded_Pattern_Path_Dict[word, talker] = self.pattern_Path_Dict[word, talker]
            for word in word_List:    #Talker exclusion
                self.excluded_Pattern_Path_Dict[word, self.excluded_Talker] = self.pattern_Path_Dict[word, self.excluded_Talker]
            return

        raise ValueError("Unsupported pattern exclusion mode")
        
    #Pattern making and inserting to queue 여기부터
    def Pattern_Generate(self):
        if pattern_Parameters.Pattern_Use_Bit == 16:
            float_Bit_Type = np.float16
            int_Bit_Type = np.int16
        elif pattern_Parameters.Pattern_Use_Bit == 32:
            float_Bit_Type = np.float32
            int_Bit_Type = np.int32
        
        #Queue
        epoch = self.start_Epoch
        while True:
            if epoch < model_Parameters.Max_Epoch_with_Exclusion:
                pattern_Path_List = [x for x in self.training_Pattern_Path_Dict.values()]
            elif epoch < model_Parameters.Max_Epoch_without_Exclusion:
                pattern_Path_List = [x for x in self.pattern_Path_Dict.values()]
            else:
                self.is_Finished = True
                break
            shuffle(pattern_Path_List)

            pattern_Path_Batch_List = [pattern_Path_List[x:x + model_Parameters.Batch_Size] for x in range(0, len(pattern_Path_List), model_Parameters.Batch_Size)]

            current_Index = 0
            is_New_Epoch = True
            while current_Index < len(pattern_Path_Batch_List):                
                if len(self.pattern_Queue) >= model_Parameters.Max_Queue:
                    time.sleep(0.1)
                    continue

                pattern_Count = len(pattern_Path_Batch_List[current_Index])

                acoustic_Pattern_List = []
                semantic_Pattern_List = []
                cycle_List = []
                for pattern_Path in pattern_Path_Batch_List[current_Index]:
                    with open(os.path.join(pattern_Parameters.Pattern_Path, pattern_Path).replace("\\", "/"), "rb") as f:
                        pattern_Dict = pickle.load(f)

                    acoustic_Pattern_List.append(pattern_Dict["Acoustic"])
                    semantic_Pattern_List.append(pattern_Dict["Semantic"])
                    cycle_List.append(pattern_Dict["Acoustic"].shape[0])

                max_Cycle = max(cycle_List)

                acoustic_Batch_Pattern = np.zeros(shape=(pattern_Count, max_Cycle, self.acoustic_Size), dtype=float_Bit_Type)
                semantic_Batch_Pattern = np.zeros(shape=(pattern_Count, max_Cycle, self.semantic_Size), dtype=float_Bit_Type)   #For sentence support, semantic is also use cycle.
                cycle_Batch_Pattern = np.stack(cycle_List).astype(int_Bit_Type)

                for pattern_Index, (acoustic_Pattern, semantic_Pattern, cycle) in enumerate(zip(acoustic_Pattern_List, semantic_Pattern_List, cycle_List)):
                    acoustic_Batch_Pattern[pattern_Index, :cycle] = acoustic_Pattern
                    semantic_Batch_Pattern[pattern_Index, :cycle] = semantic_Pattern

                self.pattern_Queue.append([
                    epoch,
                    is_New_Epoch,
                    {
                        self.placeholder_Dict["Is_Training"]: True,
                        self.placeholder_Dict["Acoustic"]: acoustic_Batch_Pattern,
                        self.placeholder_Dict["Semantic"]: semantic_Batch_Pattern,
                        self.placeholder_Dict["Length"]: cycle_Batch_Pattern,
                        }
                    ])
                current_Index += 1
                is_New_Epoch = False
                
            epoch += 1

    #Pop a training pattern
    def Get_Pattern(self):
        while len(self.pattern_Queue) == 0: #When training speed is faster than making pattern, model should be wait.
            time.sleep(0.01)
        return self.pattern_Queue.popleft()
    
    #This function will be called only one time.
    def Test_Pattern_Generate(self):
        if pattern_Parameters.Pattern_Use_Bit == 16:
            float_Bit_Type = np.float16
            int_Bit_Type = np.int16
        elif pattern_Parameters.Pattern_Use_Bit == 32:
            float_Bit_Type = np.float32
            int_Bit_Type = np.int32
        else:
            assert False

        self.test_Pattern_Dict = {
            "Count": len(self.pattern_Path_Dict),
            "Max_Cycle": max([cycle for cycle in self.cycle_Dict.values()])
            }
        self.test_Pattern_Dict["Index_Dict"] = {}
        self.test_Pattern_Dict["Acoustic_Pattern"] = np.zeros((self.test_Pattern_Dict["Count"], self.test_Pattern_Dict["Max_Cycle"], self.acoustic_Size)).astype(float_Bit_Type)
        self.test_Pattern_Dict["Semantic_Pattern"] = np.zeros((self.test_Pattern_Dict["Count"], self.test_Pattern_Dict["Max_Cycle"], self.semantic_Size)).astype(float_Bit_Type)
        self.test_Pattern_Dict["Cycle_Pattern"] = np.zeros((self.test_Pattern_Dict["Count"])).astype(int_Bit_Type)

        for index, ((word, talker), pattern_Path) in enumerate(self.pattern_Path_Dict.items()):
            with open (os.path.join(pattern_Parameters.Pattern_Path, pattern_Path).replace("\\", "/"), "rb") as f:
                load_Dict = pickle.load(f)

            self.test_Pattern_Dict["Index_Dict"][word, talker] = index
            self.test_Pattern_Dict["Acoustic_Pattern"][index, :load_Dict["Acoustic"].shape[0]] = load_Dict["Acoustic"]
            self.test_Pattern_Dict["Semantic_Pattern"][index, :load_Dict["Semantic"].shape[0]] = load_Dict["Semantic"]
            self.test_Pattern_Dict["Cycle_Pattern"][index] = load_Dict["Acoustic"].shape[0]
            
        self.test_Pattern_Dict["Feed_Dict_List"] = []
        for start_Index in range(0, self.test_Pattern_Dict["Count"], model_Parameters.Batch_Size):
            test_Pattern = self.test_Pattern_Dict["Acoustic_Pattern"][start_Index:start_Index + model_Parameters.Batch_Size]
            test_Length = np.ones((test_Pattern.shape[0])) * self.test_Pattern_Dict["Max_Cycle"]
            new_Feed_Dict= {
                self.placeholder_Dict["Is_Training"]: False,
                self.placeholder_Dict["Acoustic"]: test_Pattern,
                self.placeholder_Dict["Length"]: test_Length
                }
            self.test_Pattern_Dict["Feed_Dict_List"].append(new_Feed_Dict)

        self.is_Test_Pattern_Generated = True

    #Return all patterns.
    def Get_Test_Pattern_List(self):
        while True:
            if self.is_Test_Pattern_Generated:
                return self.test_Pattern_Dict["Feed_Dict_List"]
            time.sleep(1.0)

    #Voice file -> pattern.
    #In current study, hidden analysis use this function.
    def Get_Test_Pattern_from_Voice(self, voice_File_Path_List):
        if pattern_Parameters.Pattern_Use_Bit == 16:
            pattern_Bit_Type = np.float16
        elif pattern_Parameters.Pattern_Use_Bit == 32:
            pattern_Bit_Type = np.float32
        else:
            assert False

        acoustic_List = []
        cycle_List = []
        
        for voice_File in voice_File_Path_List:
            if pattern_Parameters.Acoutsic_Mode.upper() == "Spectrogram".upper():
                sig = librosa.core.load(voice_File, sr = pattern_Parameters.Spectrogram.Sample_Rate)[0]
                sig = librosa.effects.trim(sig, frame_length = 32, hop_length=16)[0]  #Trim
                acoustic_Array = np.transpose(spectrogram(
                    sig,
                    num_freq= pattern_Parameters.Spectrogram.Dimension,
                    frame_shift_ms= pattern_Parameters.Spectrogram.Frame_Shift,
                    frame_length_ms= pattern_Parameters.Spectrogram.Frame_Length,
                    sample_rate= pattern_Parameters.Spectrogram.Sample_Rate,
                    ))
                acoustic_List
            elif pattern_Parameters.Acoutsic_Mode.upper() == "Mel".upper():
                sig = librosa.core.load(voice_File, sr = pattern_Parameters.Mel.Sample_Rate)[0]
                sig = librosa.effects.trim(sig, frame_length = 32, hop_length=16)[0]  #Trim
                acoustic_Array = np.transpose(melspectrogram(
                    sig,
                    num_freq= pattern_Parameters.Mel.Spectrogram_Dim,
                    frame_shift_ms= pattern_Parameters.Mel.Frame_Shift,
                    frame_length_ms= pattern_Parameters.Mel.Frame_Length,
                    num_mels= pattern_Parameters.Mel.Mel_Dim,
                    sample_rate= pattern_Parameters.Mel.Sample_Rate,
                    max_abs_value= pattern_Parameters.Mel.Max_Abs
                    ))
            
            acoustic_List.append(acoustic_Array)
            cycle_List.append(acoustic_Array.shape[0])

        acoustic_Pattern = np.zeros((len(acoustic_List), max(cycle_List), self.acoustic_Size)).astype(pattern_Bit_Type)
        for index, acoustic_Array in enumerate(acoustic_List):
            acoustic_Pattern[index, :cycle_List[index], :] = acoustic_Array
        cycle_Pattern = np.hstack(cycle_List).astype(pattern_Bit_Type)
        
        #Semantic pattern is not used in the test.
        new_Feed_Dict= {
            self.placeholder_Dict["Is_Training"]: False,
            self.placeholder_Dict["Acoustic"]: acoustic_Pattern,
            self.placeholder_Dict["Length"]: cycle_Pattern
            }

        return new_Feed_Dict