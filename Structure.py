#Imports
import pandas as pd
import os

#Functions
def snippetConllu2DF(conllu_lines: str):
    """
    Function for transforming a string of conllu formatted text into a dataframe usable by the other functions in this package
    """
    df = pd.DataFrame([line.split('\t') for line in conllu_lines.split('\n')])
    df.columns = ['id', 'text', 'lemma', 'upos', 'xpos', 'feats', 'head', 'deprel', 'deps', 'misc']
    
    return df.dropna()

def initBooksFromJsons(json_path: str) -> dict:
    """
    Function which takes in a path to a folder containing .json files produced by Trankit and creates python dicts from them
    :JSON_PATH: path to folder as str
    :return: python dict with form [book_name, pandas.DataFrame]
    """

    books = {}
    #Loading the conllus (jsons) as Dataframes

    for file in os.listdir(json_path):
        #Opening json contents
        with open(json_path+"/"+file) as json_file:
            #Transform into dataframe
            df = pd.read_json(json_file)
            #Append as dict juuuuust in case we need the metadata
            #Clip at 17 as the format for the filenames are standardized
            books[file[:17]] = df
    return books

def initBooksFromConllus(conllu_path: str) -> dict:
    """
    Function which takes in a path to a folder containing conllu files and returns a dict with pd.DataFrames of sentence data
    :conllu_path: path to folder of conllus as str
    :return: dict of form [book_name, pd.DataFrame], df is sentence data of a book
    """

    books = {}
    #Loading the conllus as Dataframes
    for file in os.listdir(conllu_path):
        #Opening conllus contents
        with open(conllu_path+"/"+file) as conllu_file:
            #Read lines from the conllu files
            conllu_lines = [x[:-1] for x in conllu_file if x[0] != '#' and x[0] != '\n']
            #Transform into dataframe
            df = pd.DataFrame([line.split('\t') for line in conllu_lines])
            #Set names for columns
            df.columns = ['id', 'text', 'lemma', 'upos', 'xpos', 'feats', 'head', 'deprel', 'deps', 'misc']
            #Clip at when the file type starts
            books[file[:file.find('.')]] = df
    return books

def findAgeFromID(key: str) -> str:
    "Function that returns the age information embedded in a book id"
    return key[key.find('_')+1:key.find('_')+1+key[key.find('_')+1:].find('_')]

def getAvailableAges(corpus: dict) -> list[int]:
    """
    Function which returns the ages that are currently available as sub corpora
    """
    return list(map(int,list(set([findAgeFromID(x) for x in list(corpus.keys())]))))

def filterRegisters(corpus: dict[str,pd.DataFrame], registers: list[int]) -> dict[str,pd.DataFrame]:
    """
    Function for creating a register sepcific subcorpus. Valid registers are:
    1 = Fiction
    2 = Non-fiction, non-textbook
    3 = Textbook
    You can pass as many registers as you want (any valid subset of [1,2,3])
    """

    returnable = {}
    for key in corpus:
        if int(key[-1]) in registers:
            df = corpus[key]
            returnable[key] = df
    return returnable

def getRangeSubCorp(corp: dict, num: int) -> dict:
    """
    Simple function to get sub_corpora from the whole package based on the target age, such that a book will go to +-1 range of its target age
    Naming conventions are ISBN_age-group_register, where age is an int [5,16]
    """
    sub_corp = {}
    for key in corp:
        age = int(findAgeFromID(key))
        if (num - age < 2 and num - age > -2):
            df = corp[key]
            sub_corp[key] = df
    return sub_corp

def getDistinctSubCorp(corp: dict, num: int) -> dict:
    """
    Simple function to get sub_corpora from the whole package based on the target age exactly, so eahc book will only be included once
    Naming conventions are ISBN_age-group_register, where age is an int [5,16]
    """
    sub_corp = {}
    for key in corp:
        if key.find('_'+str(num)+'_') != -1:
            sub_corp[key] = corp[key]
    return sub_corp


def combineSubCorpDicts(corps: list) -> dict:
    """
    Combine a list of sub-corp dicts into one dict
    """
    whole = corps[0].copy()
    for i in range(1, len(corps)):
        whole.update(corps[i])
    return whole

def combineSubCorpsData(corps: list, sum_together: bool):
    """
    Takes in a list of dataframes (or series) and combines them together
    """
    dfs = []
    for df in corps:
        dfs.append(df)
    combined = pd.concat(dfs)
    if sum_together:
        if type(combined) is pd.DataFrame:
            return combined.groupby(combined.columns[0])[combined.columns[1]].sum().reset_index()
        else:
            return combined.groupby(level=0).sum()
    return combined

def buildIdTreeFromConllu(conllu: pd.DataFrame) -> dict[int,list[int]]:
    """
    Build a tree for each sentence in a conllu file, where each node points to the corresponding DataFrame row of a line in the conllu-file
    """
    #Due to using properly formatted CoNLLUs, deal with 'ettei' etc. compounds by ignoring them (as they should be on the syntactic tree)
    conllu = conllu[conllu['head'] != '_']
    conllu['id'] = conllu['id'].apply(lambda x: str(x))
    id_tree = {}
    #First fetch ids marking boundaries for each sentence
    sentence_ids = []
    start = 0
    for i in range(1,len(conllu)):
        if conllu.iloc[i]['id'] == '1':
            sentence_ids.append((start,i-1))
            start = i
    sentence_ids.append((start, len(conllu)-1))
    #Build trees for each sentence
    for sentence in sentence_ids:
        root = 0
        sent_locs = range(sentence[0],sentence[1]+1)
        heads = conllu.iloc[sentence[0]:sentence[1]+1]['head'].to_numpy(int)-1
        sent_tree = {x:[] for x in sent_locs}
        for j in range(len(heads)):
            if heads[j] == -1:
                root = sent_locs[j]
            else:
                children = sent_tree[sent_locs[heads[j]]]
                children.append(sent_locs[j])
                sent_tree[sent_locs[heads[j]]] = children            
        id_tree[root] = sent_tree
    return id_tree

