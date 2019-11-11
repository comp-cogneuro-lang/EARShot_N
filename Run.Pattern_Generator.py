import Pattern_Generator
from Hyper_Parameters import pattern_Parameters
from concurrent.futures import ThreadPoolExecutor as PE
import os

max_Worker=10

file_List = []
for root, dirs, files in os.walk(pattern_Parameters.Voice_Path):
    for file in files:
        name, ext = os.path.splitext(file.upper())
        if len(name.split('_')) != 2 or ext != '.wav'.upper():
            continue
        word, talker = name.split('_')
        
        if not word in Pattern_Generator.using_Word_List:
            continue
        
        file_List.append((
            word,
            Pattern_Generator.pronunciation_Dict[word],
            talker,
            os.path.join(root, file).replace('\\', '/')
            ))
        
with PE(max_workers = max_Worker) as pe:
    for word, pronunciation, identifier, voice_File_Path in file_List:        
        pe.submit(
           Pattern_Generator.Pattern_File_Geneate,
           word,
           pronunciation,
           identifier, #In paper, identifier is 'talker'.
           voice_File_Path
           )
        # Pattern_Generator.Pattern_File_Geneate(
        #    word,
        #    pronunciation,
        #    identifier, #In paper, identifier is 'talker'.
        #    voice_File_Path
        #    )
    
Pattern_Generator.Metadata_Generate()
#Pattern_Generator.Metadata_Subset_Generate(identifier_List = ["EO", "JM"], metadata_File_Name= "Metadata.Human.pickle")

#with open('list240.txt', 'r') as f:
#    list_240 = [x.strip() for x in f.readlines()]
#Pattern_Generator.Metadata_Subset_Generate(word_List= list_240, metadata_File_Name= "Metadata.240.pickle")

# Pattern_Generator.Metadata_Subset_Generate(identifier_List = ["Agnes", "Alex", "Bruce", "Fred", "Junior", "Kathy", "Princess", "Ralph", "Vicki", "Victoria"], metadata_File_Name= "Metadata.10Talkers.pickle")

