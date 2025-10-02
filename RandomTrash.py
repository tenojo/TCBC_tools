from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import Structure
import FeatureExtraction as fe

#These might get some use some day

def getBookLemmaCosineSimilarities(corpus: dict[str,pd.DataFrame], f_lemma: pd.Series) -> pd.DataFrame:
    """
    Calculating cosine similarities of all lemmas between the books in the corpus. Inspired by Korochkina et el. 2024
    """
    tf_idf_scores = {}

    #Sort the books so that we get groupings by age group
    sorted_keys = list(corpus.keys())
    sorted_keys.sort(key=lambda x:int(Structure.findAgeFromID(x)))

    #Get all corpus' lemmas from lemma frequency data
    all_lemmas = list(f_lemma.index)
    book_vectorizer = TfidfVectorizer(vocabulary=all_lemmas)
    for book in sorted_keys:
        #Tf-idf scores from lemma data of a book
        book_lemmas = " ".join(corpus[book]['lemma'].to_numpy(dtype=str))
        #print(book_lemmas.values)
        tf_idf_scores[book] = book_vectorizer.fit_transform([book_lemmas])
    similarity_scores = {}
    for book in sorted_keys:
        #Compare current book to every other book
        scores = []
        for comp in sorted_keys:
            scores.append(cosine_similarity(tf_idf_scores[book], tf_idf_scores[comp]))
        similarity_scores[book] = scores
    #Create df
    matrix_df = pd.DataFrame.from_dict(similarity_scores, orient='index').transpose()
    #Set indexes correctly
    matrix_df.index = tf_idf_scores.keys()
    #Dig out the values from nd.array
    matrix_df_2 = matrix_df.copy().applymap(lambda x: x[0][0])
    return matrix_df_2

# Lol


def getOnlyAlnums(sentences: dict, column: str) -> dict:
    """
    Function which takes in sentence data and cleans punctuation and other non-alnum characters
    :param sentences:dict of form [book_name, pd.DataFrame], df is sentence data
    :param column: name of the column which to clean (recommend 'text' or 'lemma')
    :return: dict of the same form
    """
    clean = {}
    for key in sentences:
        df = sentences[key]
        #Get rid of PUNCT
        no_punct = df[df.upos != "PUNCT"].copy()
        #Make words lowercase
        no_punct[column] = no_punct[column].apply(lambda x: x.lower())
        #Remove non-alnums
        no_punct[column] = no_punct[column].apply(lambda x: ''.join(filter(str.isalnum, x)))
        #Filter rows with nothing in them
        no_punct = no_punct[no_punct.text != '']
        clean[key] = no_punct
    return clean



#Get PoS frequencies
def getPOSFrequencies(sentences: dict, scaler_sentences: bool=None) -> dict:
    """
    Function which gets the POS frequencies of sentences of books
    """
    pos_freqs = {}

    if scaler_sentences:
        sentences_sizes = fe.getNumOfSentences(sentences)

    for key in sentences:
        #Map book_name to pivot table
        if scaler_sentences:
            freqs = sentences[key]['upos'].value_counts()
            pos_freqs[key]=freqs/sentences_sizes[key]
        else:
            pos_freqs[key] = sentences[key]['upos'].value_counts()
        #pd.DataFrame.pivot_table(sentences[key], columns='upos', aggfunc='size').sort_values(ascending=False).reset_index().rename(columns={0: "frequency"})

    return pos_freqs


#Functions to get metrics from sentences

#Function the get the average length of the unique lemmas in the sentenes
def getAvgLen(data: dict, column: str=None) -> dict:
    """
    Get the average length of either words or lemmas from sentence data. Works for both original sentence data (pd.DataFrame) and processed ones,
    such as frequency data (pd.Series)
    :data: dict of form [book_name, pd.DataFrame], df should contain sentence data
    :column: name of the CoNLLU column for which to calculate the average. Recommend either 'text' for words and 'lemma' for lemmas
    :return: dict of [book_name, avg_len], where avg_len is float
    """
    avg_lens = {}
    for key in data:
        i = 1
        total_len = 0
        df = data[key]
        if type(df) is pd.DataFrame:
            #For each lemma count the length and add one to counter
            for lemma in df[column]:
                #Only care about strings
                if type(lemma) is str:
                    total_len += len(lemma)
                    i += 1
        elif type(df) is pd.Series:
            #For each lemma count the length and add one to counter
            for lemma in list(df.index):
                #Only care about strings
                if type(lemma) is str:
                    total_len += len(lemma)
                    i += 1
        #If no lemmas were found (should never happen but just in case), we make the avg_len be 0
        if i==1:
            avg_lens[key] = 0
        else:
            #Map book_name to avg lemma length
            avg_lens[key] = total_len/(i-1.0)
    return avg_lens



#Function to calculate DP (deviation of proportions) of all the words in the corpus


#Function to calculate D (dispersion) of all the words in the corpus





#Functions to do with sub-corpora




    





def cleanLemmas(sentences: dict) -> dict:
    """
    Function which takes in sentence data and cleans rows based on a number of filters on lemma forms to guarantee better quality data
    Does get rid of PUNCT
    :param sentences:dict of form [book_name, pd.DataFrame], df is sentence data
    :return: dict of the same form, but better data
    """
    clean = {}
    for key in sentences:
        df = sentences[key]
        no_punct = df.copy()

        #Make words lowercase
        no_punct['lemma'] = no_punct['lemma'].apply(lambda x: str(x).lower())
        #First mask
        #Remove lemmas which are not alnum or have '-' but no weird chars at start or end, length >1, has no ' ', and has no ','
        m = no_punct.lemma.apply(lambda x: (x.isalnum() 
                                            or (not x.isalnum() and '-' in x and x[0].isalnum() and x[len(x)-1].isalnum())
                                            or (not x.isalnum() and '#' in x and x[0].isalnum() and x[len(x)-1].isalnum())
                                            and len(x)>1 
                                            and not ' ' in x
                                            and not ',' in x))
        filtered = no_punct[m]
        #Second mask
        #Remove lemmas that have the same character more than thrice consecutively at the start (Finnish doesn't work like this)
        m_2 = no_punct.lemma.apply(lambda x: conseqChars(x)
                                   and not (x.isnumeric() and len(x)>4)
                                   )
        filtered_2 = filtered[m_2] 
        clean[key] = filtered_2
    return clean

def conseqChars(x: str):
    if len(x)>2:
        return not x[0]==x[1]==x[2]
    else:
        return True
    
def cleanWords(books: dict) -> dict:
    """
    Function for cleaning non-alnum characters from the beginning and ending of words and lemmas in sentence data
    :param books: dict of the sentence data of books
    :return: dictionary, where the dataframes have been cleaned
    """
    #Clean words
    clean = {}
    for key in books:
        df = books[key].copy()
        df['text'] = df['text'].apply(lambda x: delNonAlnumStart(str(x)))
        df['lemma'] = df['lemma'].apply(lambda x: delNonAlnumStart(str(x)))
        df['text'] = df['text'].apply(lambda x: delNonAlnumEnd(str(x)))
        df['lemma'] = df['lemma'].apply(lambda x: delNonAlnumEnd(str(x)))
        clean[key] = df.dropna()
    return clean

def delNonAlnumStart(x: str) -> str:
    '''
    Function for deleting non-alnum sequences of words from Conllu-files
    :param x: string that is at least 2 characters long
    :return: the same string, but with non-alnum characters removed from the start until the first alnum-character
    '''
    if not x[0].isalnum() and len(x)>1:
        ind = 0
        for i in range(len(x)):
            if x[i].isalnum():
                ind=i
                break
        return x[ind:]
    return x    

def delNonAlnumEnd(x: str) -> str:
    '''
    Function for deleting non-alnum sequences of words from Conllu-files
    :param x: string that is at least 2 characters long
    :return: the same string, but with non-alnum characters removed from the start until the first alnum-character
    '''
    if not x[-1].isalnum() and len(x)>1:
        ind = 0
        for i in range(1,len(x)):
            if x[-i].isalnum():
                ind=-i
                break
        return x[:ind+1]
    return x 

def ignoreOtherAlphabets(sentences: dict) -> dict:
    """
    Function which takes in sentence data and cleans rows based on if words contain characters not in the Finnish alphabe (e.g. cyrillic)
    :param sentences:dict of form [book_name, pd.DataFrame], df is sentence data
    :return: dict of the same form, but better data
    """
    clean = {}
    for key in sentences:
        df = sentences[key]
        no_punct = df.copy()

        #Make words lowercase
        no_punct['text'] = no_punct['text'].apply(lambda x: str(x).lower())
        #First mask
        #Remove words which are not in the Finnish alphabet
        m = no_punct.text.apply(lambda x: (
                len(x.encode("ascii", "ignore")) == len(x)
                or x.find('ä') != -1
                or x.find('ö') != -1 
                or x.find('å') != -1
            )
        )
        filtered = no_punct[m]
        clean[key] = filtered
    return clean












def getDictAverage(corp_data: dict) -> float:
    """
    Simple function for calculating the average value of a dict containing book ids and some numerical values
    """
    return sum(list((corp_data.values())))/len(list(corp_data.keys()))



def addAgeGroupSeparatorsToDF(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function for adding separator lines to a df that's meant to be shown as a heatmap!
    """
    indices = list(df.index)
    one2two = 0
    while int(Structure.findAgeFromID(indices[one2two]))<9:
        one2two += 1
    two2three = one2two
    while int(Structure.findAgeFromID(indices[two2three]))<13:
        two2three += 1
    df.insert(one2two, 'one2two', pd.Series([1]*len(indices)))
    df.insert(two2three+1, 'two2three', pd.Series([1]*len(indices)))
    temp_dict = dict(zip(df.columns, ([1]*(len(indices)+2))))
    row1 = pd.DataFrame(temp_dict, index=['one2two'])
    row2 = pd.DataFrame(temp_dict, index=['two2three'])
    df_2 = pd.concat([df.iloc[:one2two], row1, df.iloc[one2two:]])
    df_2 = pd.concat([df_2.iloc[:two2three+1], row2, df_2.iloc[two2three+1:]])
    return df_2

def combineSeriesForExcelWriter(f_lemmas, corpus, lemma_DP, lemma_CD, lemma_zipfs, f_words, word_DP, word_CD, word_zipfs, pos):
    """
    Helper function for combining various Series containing lemma/word data into compact dataframes
    """
    lemma_data = pd.concat([f_lemmas, fe.getTaivutusperheSize(corpus), lemma_DP, lemma_CD], axis=1)
    lemma_data.columns = ['frequency','t_perh_size', 'DP', 'CD']

    word_data = pd.concat([f_words, word_DP, word_CD], axis=1)
    word_data.columns = ['frequency', 'DP', 'CD']


    return lemma_data, word_data




#Moving to a regression task instead of hard age groups



def mapExactAgeToMean(corpus: dict[str,pd.DataFrame]) -> dict[str,pd.DataFrame]:
    """
    Function for taking exact ages in ids and mapping them to age groups/means of age intervals
    """
    returnable = {}
    for key in corpus:
        if int(Structure.findAgeFromID(key))<9:
            new_key = key[:key.find('_')]+'_7_'+key[-1]
        elif 8<int(Structure.findAgeFromID(key))<13:
            new_key = key[:key.find('_')]+'_10_'+key[-1]
        else:
            new_key = key[:key.find('_')]+'_14_'+key[-1]
        df = corpus[key]
        returnable[new_key] = df
    return returnable
