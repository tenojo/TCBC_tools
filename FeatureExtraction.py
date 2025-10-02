#Imports
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
import random
from collections import Counter
import string
import itertools

#Section for syllable counting and what is necessary for that

CONSONANTS = ['b','c','d','f','g','h','j','k','l','m','n','p','q','r','s','t','v','w','x','z']
VOCALS = ['a','e','i','o','u','y','å','ä','ö']
DIFTONGS_1 = ['ai','ei','oi','ui','yi','äi','öi','ey','iy','äy','öy','au','eu','iu','ou', 'aa', 'ee', 'ii', 'oo', 'uu', 'yy', 'ää', 'öö']
DIFTONGS_2 = ['ie','uo','yö']
SYLL_SET_1 = set(''.join(i) for i in itertools.product(CONSONANTS, VOCALS))
SYLL_SET_2 = set(''.join(i) for i in itertools.product(VOCALS, VOCALS)) - set(DIFTONGS_1+DIFTONGS_2)
SYLL_SET_3 = set(DIFTONGS_2)

def countSyllablesFinnish(word: str) -> int:
    """
    Function for calculating syllables of words in Finnish.
    Is not perfect, as cases like "Aie" will get marked as having one syllable instead of two, 
    but these edge cases are very hard to code into rules and due to limited resources, have been left 'unfinished'
    """
    syll_count = 0
    first_round_word = ""

    word = str(word).capitalize()
    #If more than one char AND does not start with punctuation/numerals
    if len(word) > 1 and not word[0] in string.punctuation and not word[0].isnumeric():
        temp_ind = 0
        for i in range(1,len(word)):
            bigram = word[i-1]+word[i]
            #If consonant+vowel
            if bigram in SYLL_SET_1:
                first_round_word += word[temp_ind:i-1]+"#"
                temp_ind = i-1
        first_round_word += word[temp_ind:]
    if first_round_word.count('#') > 0:
        candidates = first_round_word.split('#')
        for i in range(len(candidates)):
            candidate = candidates[i]
            if len(candidate) == 0:
                continue
            syll_count += 1
            for j in range(1,len(candidate)):
                bigram = candidate[j-1]+candidate[j]
                #If not recognized diftong
                if bigram in SYLL_SET_2:
                    syll_count += 1
                #If special diftong that counts as a syllable after the first syllable
                if i!=0 and bigram in SYLL_SET_3:
                    syll_count += 1
    elif len(word) == 0:
        return 0
    #If comprises of only one character and not punct/sym/num
    elif not word[0] in string.punctuation and not word[0].isnumeric():
        syll_count += 1
    return syll_count

def getSyllableAmountsForWords(words: pd.Series) -> pd.Series:
    "Helper function to apply syllable counting to all words"
    uniq_words = words.drop_duplicates()
    return pd.Series(data=uniq_words.apply(countSyllablesFinnish).to_numpy(), index=uniq_words.to_numpy(dtype='str'))

def getAverageSyllablesPerSentence(conllu: pd.DataFrame, id_tree: dict[int, dict[int, list[int]]]):
    """
    Calculate the average amount of syllables per sentence (total number of syllables / number of sentences)
    """
    syllables = np.sum([countSyllablesFinnish(x) for x in conllu['text'].to_numpy(str)])
    return syllables / len(id_tree)

#Functions

def getColumnFrequencies(corpus: dict[str,pd.DataFrame], columns=list[str]) -> dict[str,pd.DataFrame]:
    """
    A more general function for calculating frequencies of rows in our corpus format
    """
    freqs = {}
    for key in corpus:
        book = corpus[key]
        freqs[key] = book[columns].value_counts()
    return freqs

def getTokenAmounts(corpus: dict) -> dict:
    """
    Get amount of tokens (aka rows) for each book
    """
    word_amounts = {}
    for key in corpus:
        df = corpus[key]
        word_amounts[key] = len(df)
    return word_amounts

def getAvgSentenceLens(corpus: dict) -> dict:
    """
    Functon for getting the average length of sentences in each book
    :param corpus: dict of form [id, pd.DataFrame] like in the other methods
    :return: dict of form [id, double], where the double is the average sentence length of the corresponding book
    """
    lens = {}
    for key in corpus:
        df = corpus[key]
        help1 = len(df)
        nums = df.id.value_counts()
        num_of_sents = nums.iloc[0]
        lens[key] = (help1/num_of_sents)
    return lens

def getDP(v: dict, f_series: pd.Series, s: dict) -> tuple:
    """
    Function which calculates the dispersion (DP) based on the formula by Gries
    :v: dict of form [book_name, pd.Series], series has frequencies per book
    :f_df: pd.Series that includes the total frequencies of words/lemmas in the whole corpus
    :s: dict of form [book_name, ratio], where ratio is how much of the whole corpus a book takes
    :return: tuple, where the first member is a pd.Series with DP, the second is a series with DP_norm
    """
    #First get the minimum s
    min_s = 1
    for key in s:
        if s[key] < min_s:
            min_s = s[key]
    #For corpus parts that are length 1
    if min_s == 1:
        min_s = 0

    words = [x[0] for x in f_series.index]
    DP = []
    DP_norm = []
    with tqdm(range(len(f_series)), desc="DP calculations") as pbar:
        #Loop through every single word in the corpus
        for word in words:
            #Get the freq of the word in the whole corpus
            f = f_series[word]
            #Calculations according to the dispersion measure by Gries 2020
            abs_sum_i = [((v[key].get(word, 0.0))/f)-s[key] for key in v]
            #Calculate and append DP
            dp = np.sum(np.absolute(abs_sum_i))*0.5
            DP.append(dp)
            #Append DP_norm to list (alltho with how many documents we have, the normalization doesn't work very well at all)
            DP_norm.append(dp/(1-min_s))
            #Update pbar
            pbar.update(1)
    return pd.Series(DP, words), pd.Series(DP_norm, words)

def getDispersion(v: dict, f_series: pd.Series) -> pd.Series:
    """
    Function which calculates the dispersion of a word in a (sub)corpus
    :v: dict of form [book_name, pd.Series], series has frequencies per book
    :f_df: pd.Series that includes the total frequencies of words/lemmas in the whole corpus
    :return: pd.Series, with words/lemmas as indices and dispersions as values
    """

    words = [x[0] for x in f_series.index]
    #v_prepped = {key:pd.Series(data=np.multiply(v[key].to_numpy(), np.log(v[key].to_numpy())),index=v[key].index) for key in v}
    mass_frame = pd.concat([pd.Series(data=np.multiply(v[key].to_numpy(), np.log(v[key].to_numpy())),index=v[key].index) for key in v]).groupby(level=0).sum().fillna(0)
    corp_len = np.log(len(list(v.keys())))
    D = []
    with tqdm(range(len(f_series)), desc="D calculations") as pbar:
        #Loop through every single word in the corpus
        for word in words:
            #Get the freq of the word in the whole corpus
            f = f_series[word]
            # D = [log(p) * sum(p_i*log(p_i))/p]/log(n)
            #Calculate and append D
            D.append((np.log(f) - (mass_frame[word] / f))/corp_len)
            #Update pbar
            pbar.update(1)
    return pd.Series(D, words)

def getU(v: dict, f: pd.Series, D: pd.Series) -> pd.Series:
    """
    Function for calculating the Estimated frequency per 1 million words (U) for words/lemmas
    """
    corp_size = f.sum()
    base_scaler = 1000000 / corp_size
    words = [x[0] for x in f.index]
    #To speed up calculations, prepare everything that we can with numpy operations before doing anything in a for-loop
    #Here we calculate the f_min value for each word (frequency of word inside a book * total length of said book)
    v_prepped = {key:pd.Series(data=np.multiply(v[key].to_numpy(), v[key].sum()),index=v[key].index) for key in v}
    #Sum all f_mins together (essentailly flatten the dictionary)
    f_mins = list(v_prepped.values())[0]
    if len(v) != 1:
        for i in range(1, len(v_prepped)):
            f_mins = f_mins.add(list(v_prepped.values())[i], fill_value=0)
    #Scale with 1/N as per formula
    f_mins = f_mins * (1/corp_size)
    #Calculate U-values for each word
    U_data = [base_scaler * (f[word]*D.get(word, 0.0) + (1-D.get(word, 0.0)) * f_mins[word]) for word in words]
    return pd.Series(U_data, index=words)

def getSFI(U: pd.Series) -> pd.Series:
    """
    Function for calculating the Standardized Frequency Index (SFI) for the estimated frequency per 1 million words (U)
    """
    values = 10*(np.log10(U.to_numpy())+4)
    return pd.Series(values, U.index)

def getCD(v: dict) -> pd.Series:
    """
    Function which gets the contextual diversity of words/lemmas based on frequency data
    """
    #Get number of books
    books_num = len(v.keys())
    word_series = []
    #For each series attached to a book, look for a frequency list and gather all the words in a list
    for key in v:
        v_series = v[key]
        word_series.append([x[0] for x in v_series.index])
    #Add all words to a new series
    series = pd.Series(word_series)
    #Create series to count in how many books does a word appear in (explode the series comprised of lists)
    CD_raw = series.explode().value_counts()
    #Return Contextual Diversity by dividing the number of appearances by the total number of books
    return CD_raw/books_num

def getL(word_amounts: dict) -> int:
    """
    Function for getting the total length of the corpus in terms of the number of words
    """
    l = 0
    for key in word_amounts:
        l += word_amounts[key]
    return l

def getS(word_amounts: dict, l: int) -> dict:
    """
    Function for getting how big each part is in relation to the total size of the corpus
    """
    s = {}
    for key in word_amounts:
        s[key] = (word_amounts[key]*1.0)/l
    return s


def combineFrequencies(freq_data: dict) -> pd.Series:
    """
    Get the total frequencies of passed freq_data in the corpus
    """
    series = []
    #Add all series to list
    for key in freq_data:
        series.append(freq_data[key])
    #Concat all series together
    ser = pd.concat(series)
    to_return = ser.groupby(ser.index).sum()
    #Retain multi-indexes if used more than one column in getting frequency data
    if type(to_return.index[0]) == tuple:
        to_return.index = pd.MultiIndex.from_tuples(to_return.index)
    #Return a series containing text as index and total freq in collection in the other
    return to_return

def getTypeTokenRatios(v: dict, word_amounts: dict) -> pd.Series:
    """
    Function which gets the type-token ratios of each book that's in the corpus
    :param v:frequency data per book
    :param word_amounts:token amounts per book
    :return: pd.Series with book names being indexes and ttr being values 
    """
    names = []
    ttrs = []
    for key in word_amounts:
        v_df = v[key]
        #Get the number of unique entities in freq data
        types = len(v_df)
        #Get the number of token in book
        tokens = word_amounts[key]
        #Add ttr to lis
        ttrs.append(types/tokens)
        #Add key to list
        names.append(key)
    return pd.Series(ttrs, names)

def getZipfValues(l: int, f: pd.Series) -> pd.Series:
    """
    Function for calculating the Zipf values of words/lemmas in a corpus
    Zipf = ( (raw_freq + 1) / (Tokens per million + Types per million) )+3.0
    :param l: total length of corpus (token amount)
    :param f: series containing frequency data of words/lemmas for the corpus
    :return: pd.Series, where indexes are words/lemmas and values the Zipf values
    """
    indexes = f.index
    types_per_mil = len(indexes)/1000000
    tokens_per_mil = l/1000000
    zipfs = f.values+1
    zipfs = zipfs / (tokens_per_mil + types_per_mil)
    zipfs = np.log10(zipfs)
    zipfs = zipfs + 3.0
    #zipfs_ser = pd.Series(zipfs, indexes)
    return pd.Series(zipfs, indexes)

def getSharedWords(wordFrequencies1: dict, wordFrequencies2: dict) -> pd.Series:
    """
    Gives a pd.DataFrame object where there are two columns: first contains those words/lemmas which are shared and the second their combined frequencies
    """
    sub1 = combineFrequencies(wordFrequencies1)
    sub2 = combineFrequencies(wordFrequencies2)

    shared = pd.concat([sub1, sub2])
    mask = shared.index.duplicated(keep=False)
    shared = shared[mask]
    return shared.groupby(shared.index).sum()

def getTaivutusperheSize(corpus: dict) -> pd.Series:
    """
    Returns a series that contains the unique lemmas of the corpus and their 'inflection family size' (taivutusperheen koko)
    """
    #First, combine the data of separate books into one, massive df
    dfs = []
    for book in corpus:
        dfs.append(corpus[book])
    #Then limit to just words and lemmas
    combined_df = pd.concat(dfs, ignore_index=True)[['lemma','feats']]
    #Drop duplicate words
    mask = combined_df.drop_duplicates()
    #Get the counts of lemmas, aka the number of different inflections
    tper = mask.value_counts('lemma')
    return tper

def dfToLowercase(df):
    """
    Simple function which maps all fields into lowercase letters
    """
    return df.copy().applymap(lambda x: str(x).lower())

def getNumOfSentences(corpus: dict) -> dict:
    """
    Function for returning the amount of main clauses for each book in the corpus
    :param corpus: Dict with form [id, pd.DataFrame]. where df has sentence data
    :return:dict of form [id, int]
    """

    sentences_sizes = {}
    for id in corpus:
        book = corpus[id]
        #Each sentences should have one word with deprel=='root' means the start of a new sentence (lause)
        sentences_sizes[id] = len(book[book['deprel'].astype(str)=='root'])
    return sentences_sizes

def getConjPerSentence(corpus: dict) -> dict:
    """
    Function for calculating the conjuction-to-sentence ratio for each book in the corpus
    :param corpus: Dict with form [id, pd.DataFrame]. where df has sentence data
    :return:dict of form [id, int]
    """
    conj_sentences_ratio = {}
    sentences_sizes = getNumOfSentences(corpus)
    for id in corpus:
        book = corpus[id]
        conj_num = len(book[book['upos'] == ('CCONJ' or 'SCONJ')])
        conj_sentences_ratio[id] = conj_num/sentences_sizes[id]
    return conj_sentences_ratio

def getPosFeaturePerBook(corpus: dict, feature: str, scaler_sentences: bool=None) -> dict:
    """
    Function for calculating the amount of wanted POS features for each book in the corpus
    :param corpus: Dict with form [id, pd.DataFrame]. where df has sentence data
    :param feature: str that maps to some dependency relation in the CoNLLU format (https://universaldependencies.org/u/dep/index.html)
    :param scaler_sentences: optional bool that forces the use of sentence amounts for scaling
    :return:dict of form [id, float]
    """

    returnable = {}
    if scaler_sentences:
        sentences_sizes = getNumOfSentences(corpus)
    for key in corpus:
        book = corpus[key]
        if scaler_sentences:
            num = len(book[book['upos'] == feature])/sentences_sizes[key]
        else:
            num = len(book[book['upos'] == feature])
        returnable[key] = num
    return returnable

def getDeprelFeaturePerBook(corpus: dict, feature: str, scaler_sentences: bool=None) -> dict:
    """
    Function for calculating the amount of wanted deprel features words for each book in the corpus
    :param corpus: Dict with form [id, pd.DataFrame]. where df has sentence data
    :param feature: str that maps to some dependency relation in the CoNLLU format (https://universaldependencies.org/u/dep/index.html)
    :param scaler_sentences: optional bool that forces the use of sentence amounts for scaling
    :return:dict of form [id, float]
    """

    returnable = {}
    if scaler_sentences:
        sentences_sizes = getNumOfSentences(corpus)
    for key in corpus:
        book = corpus[key]
        if scaler_sentences:
            num = len(book[book['deprel'] == feature])/sentences_sizes[key]
        else:
            num = len(book[book['deprel'] == feature])
        returnable[key] = num
    return returnable

def getFeatsFeaturePerBook(corpus: dict, feature: str, scaler_sentences: bool=None) -> dict:
    """
    Function for calculating the amount of wanted feats feature for each book in the corpus
    :param corpus: Dict with form [id, pd.DataFrame]. where df has sentence data
    :param feature: str that maps to some dependency relation in the CoNLLU format (https://universaldependencies.org/u/dep/index.html)
    :param scaler_sentences: optional bool that forces the use of sentence amounts for scaling
    :return:dict of form [id, float]
    """

    returnable = {}
    if scaler_sentences:
        sentences_sizes = getNumOfSentences(corpus)
    for key in corpus:
        book = corpus[key]
        #Mask those rows that don't have the wanted feature
        m = book.copy().feats.apply(lambda x: (
            x.find(feature) != -1
                )
            )
        if scaler_sentences:
            num = len(book[m])/sentences_sizes[key]
        else:
            num = len(book[m])
        returnable[key] = num
    return returnable

def getMultiVerbConstrNumPerSentence(corpus: dict) -> dict:
    """
    Function for calculating the conjuction-to-sentence ratio for each book in the corpus
    :param corpus: Dict with form [id, pd.DataFrame]. where df has sentence data
    :return:dict of form [id, int]
    """
    multiverb_sentences_ratio = {}
    sentences_sizes = getNumOfSentences(corpus)
    for id in corpus:
        book = corpus[id]
        modal_verb_num = len(book[((book['upos'] == 'AUX') & (book['xpos'] == 'V') & (book['deprel'] == 'aux')) | ((book['upos'] == 'VERB') & (book['deprel'] == 'xcomp'))])
        multiverb_sentences_ratio[id] = modal_verb_num/sentences_sizes[id]
    return multiverb_sentences_ratio

def getPreposingAdverbialClauses(corpus: dict[str,pd.DataFrame]) -> dict:
    """
    Function for calculating the number of preposing adverbial clauses in a conllu file
    """
    returnable = {}
    for key in corpus:
        df = corpus[key]
        df['head'] = df['head'].apply(lambda x: int(x))
        df['id'] = df['id'].apply(lambda x: int(x))
        prep_advcl = 0
        advcl_id = 1000
        root_id = 1000
        for i in range(len(df)):
            if df.loc[i]['id'] == 1:
                advcl_id = 1000
                root_id = 1000
            if df.loc[i]['deprel'] == 'root':
                root_id = i
            if df.loc[i]['deprel'] == 'advcl':
                advcl_id = i
                if advcl_id < root_id:
                    prep_advcl += 1
        returnable[key] = prep_advcl
    return returnable

def getPosPhraseCounts(corpus: dict[str,pd.DataFrame], upos: str) -> dict[str,float]:
    """
    Function for calculating the number of POS phrases for each book in (sub)corpus.
    Working POS-tags are those listed in the CoNLLU file format
    """
    returnable = {}
    for key in corpus:
        df = corpus[key]
        returnable[key] = len(df[(df['upos'] == upos) & (df['deprel'] == 'root')])
    return returnable

def getPosNGramForCorpus(corpus: dict[str,pd.DataFrame], n: int) -> dict[str, Counter]:
    """
    Function for getting the number of wanted length POS n-grams for each book in corpus
    """
    returnable = {}
    for key in corpus:
        df = corpus[key]['upos']
        n_grams = []
        for i in range(len(df)-(n-1)):
            n_grams += [list(df.iloc[i:i+n].to_numpy(str))]
        n_grams = map(tuple, n_grams)
        returnable[key] = Counter(n_grams)
    return returnable

def getFleschKincaidGradeLevel(corpus: dict):
    returnable = {}
    ASL = getAvgSentenceLens(corpus)
    for id in corpus:
        df = corpus[id]
        ASW = np.mean(pd.Series(data=df['text'].apply(countSyllablesFinnish).to_numpy(), index=df['text'].to_numpy(dtype='str')).to_numpy(na_value=0))
        returnable[id] = 0.39*ASL[id] + 11.8*ASW - 15.59
    return returnable

def getModifiedSmogIndexForFinnish(conllu: pd.DataFrame, id_tree: dict[int, dict[int, list[int]]]):
    """
    Calculate the SMOG (Simple Measure of Gobbledygook) index for a conllu-file.
    Using the version modified for Finnish as presented by Geoff Taylor (https://doi.org/10.1016/j.ssci.2012.01.016)
    It can only be calculated for texts that are at least 30 sentences long, so if the number of sentences is lower, we return 0.
    """
    #The modification is to increase the number of syllables from 3 in the original SMOG index to 5 for Finnish
    poly_syllable_cutoff = 5

    if len(id_tree) < 31:
        return 0
    
    smog_sentences = []
    sentences_to_pick_from = list(id_tree.keys())
    cutoff = math.floor((len(sentences_to_pick_from)/3))
    #Randomly sample 10 sentences from the beginning, middle, and end of the text
    for i in range(3):
        start_index = random.randint(0,cutoff-10)
        smog_sentences += sentences_to_pick_from[start_index:start_index+10]
        sentences_to_pick_from = sentences_to_pick_from[cutoff:]
    #Count number of polysyllable words in the sampled sentences
    num_of_polysyllables = 0
    for sentence_head in smog_sentences:
        tree = id_tree[sentence_head]
        for leaf in tree:
            if countSyllablesFinnish(conllu['text'].iloc[leaf]) > poly_syllable_cutoff-1:
                num_of_polysyllables += 1
    #Return SMOG index formula result
    return 1.0430*np.sqrt(num_of_polysyllables)+3.1291

#Coleman-Liau index
def getColemanLiauIndex(conllu: pd.DataFrame, id_tree: dict[int, dict[int, list[int]]]):
    num_of_words = len(conllu)
    num_of_sentences = len(id_tree)
    num_of_letters = np.sum(list(map(len, conllu['text'].to_numpy(str))))
    return 0.0588 * (num_of_letters / num_of_words) * 100 - 0.296 * (num_of_sentences / num_of_words) * 100 - 15.8

#Automated readability index
def getARI(conllu: pd.DataFrame, id_tree: dict[int, dict[int, list[int]]]):
    num_of_words = len(conllu)
    num_of_sentences = len(id_tree)
    num_of_letters = np.sum(list(map(len, conllu['text'].to_numpy(str))))
    return 4.71 * (num_of_letters / num_of_words) + 0.5 * (num_of_words / num_of_sentences) - 21.43

def cohensdForSubcorps(subcorp1: dict, subcorp2: dict) -> float:
    """
    Function for calculating the effect size using Cohen's d for some feature values of two subcorpora
    :param subcorp1: dictionary of form [id, float], calculated with e.g. getDeprelFeaturePerBook()
    :param subcorp2: dict of the same form as above
    :return: flaot measuring the effect size
    """
    data1 = list(subcorp1.values())
    data2 = list(subcorp2.values())
    #Sample size
    n1, n2 = len(data1), len(data2)
    #Variance
    s1, s2 = np.var(data1, ddof=1), np.var(data2, ddof=1)
    #Pooled standard deviation
    s = math.sqrt( ((n1-1)*s1 + (n2-1)*s2)/(n1+n2-2) )
    #Return Cohen's d
    return ((np.mean(data1)-np.mean(data2)) / s)

def getDepthOfTree(head:int, tree: dict[int,list[int]], depth=0) -> int:
    """
    Get syntactic tree depth recursively
    """
    next = depth+1
    children = tree[head]
    if len(children) > 0:
        for child in children:
            depth = max(depth, getDepthOfTree(child, tree, next))
    return depth


def getMeanSyntacticTreeDepth(tree):
    """
    Get the average (mean) depth of the syntactic tree of a conllu-file
    """
    depths = []
    for head in tree:
        depths.append(getDepthOfTree(head, tree[head]))
    return np.mean(depths)

def getMaxSyntacticTreeDepth(tree):
    """
    Get the average (mean) depth of the syntactic tree of a conllu-file
    """
    depths = []
    for head in tree:
        depths.append(getDepthOfTree(head, tree[head]))
    return max(depths)

def getIdTreeNGram(tree, prev_round=[]):
    """
    Recursively add layers to the n-grams.
    Returns a list of lists, which consist of ids. Amount per list is the specified n.
    """
    current_round = []
    for gram in prev_round:
        if len(tree[gram[-1]]) > 0:
            for leaf in tree[gram[-1]]:
                current_round.append(gram+[leaf])
    return current_round

def getSyntacticTreeNGram(conllu: pd.DataFrame, id_tree: dict[int, dict[int, list[int]]], n: int, feature_column: str):
    """
    Function for getting n-grams from a conllu-file of the wanted feature (UD output) column
    Most often you will have either 'deprel' for dependency relations or 'upos' for POS tags
    Expects you to have built the id-tree beforehand using buildIdTreeFromConllu()
    Returns the n-grams as a dictionary of tuple-count pairs
    """
    feat_col = conllu[feature_column]
    all_id_grams = []
    for root in id_tree:
        tree = id_tree[root]
        init_grams = [[x] for x in list(tree.keys())]
        for i in range(1, n):
            init_grams = getIdTreeNGram(tree, init_grams)
        all_id_grams += init_grams
    all_n_grams = map(tuple, [[feat_col[y] for y in x] for x in all_id_grams])
    return Counter(all_n_grams)

def findNestingSubclauses(conllu: pd.DataFrame, id_tree: dict[int, dict[int, list[int]]]):
    """
    Finds the subordinate clauses that have nesting clauses (clause within a clause)
    Only looks to find at least one nesting clause and return a list of dicts with the following keys:
    sentence_head:id of sentence head, clause_head:id of clause with nesting clauses, clause_type:deprel type of head 
    """
    deprel_conllu = conllu['deprel']
    clauses = ['csubj', 'ccomp', 'xcomp', 'advcl', 'acl', 'acl:relcl', 'xcomp:ds']
    nesting_clauses = []
    #Check children of sentence heads
    for head in id_tree:
        tree = id_tree[head]
        head_children = tree[head]
        for child in head_children:
            child_deprel = deprel_conllu.iloc[child]
            #If children deprel is a clause
            if child_deprel in clauses:
                #Check all grandchildren of the child that is a clause
                for grandchild in tree[child]:
                    grandchild_deprel = deprel_conllu.iloc[grandchild]
                    #If a nesting clause is found, append child's data to list and move to the next child
                    if grandchild_deprel in clauses:
                        nesting_clauses.append({'sentence_head':head, 'clause_head':child, 'clause_type':child_deprel})
                        break
                
    return nesting_clauses

def findStackingClauses(conllu: pd.DataFrame, id_tree: dict[int, dict[int, list[int]]]):
    """
    Finds the subordinate clauses that have stacking clauses (coordinating clause within subordinating clause)
    Only looks to find at least one stacking clause and return a list of dicts with the following keys:
    sentence_head:id of sentence head, clause_head:id of clause with stacking clauses, clause_type:deprel type of head 
    """
    deprel_conllu = conllu['deprel']
    clauses = ['csubj', 'ccomp', 'xcomp', 'advcl', 'acl', 'acl:relcl', 'xcomp:ds']
    coordinating = 'conj'
    stacking_clauses = []
    #Check children of sentence heads
    for head in id_tree:
        tree = id_tree[head]
        head_children = tree[head]
        for child in head_children:
            child_deprel = deprel_conllu.iloc[child]
            #If children deprel is a clause
            if child_deprel in clauses:
                #Check all grandchildren of the child that is a clause
                for grandchild in tree[child]:
                    grandchild_deprel = deprel_conllu.iloc[grandchild]
                    #If a stacking clause is found, append child's data to list and move to the next child
                    if grandchild_deprel == coordinating:
                        stacking_clauses.append({'sentence_head':head, 'clause_head':child, 'clause_type':child_deprel})
                        break
    
    return stacking_clauses
                

def getNonClausalChildrenAmount(deprel_conllu: pd.Series, tree: dict[int, list[int]], head):
    """
    Helper function for 'findMeanLengthOFClause'
    """
    clauses = ['csubj', 'ccomp', 'xcomp', 'advcl', 'acl', 'acl:relcl', 'xcomp:ds']
    leaves = tree[head]
    non_clausal = [head]
    #While there are leaves yet to be explored
    while len(leaves) > 0:
        #Pop first member and select it
        leaf = leaves.pop(0)
        #If leaf starts a new clause, move on
        if deprel_conllu.iloc[leaf] in clauses:
            continue
        #If leaf did not start a new clause, add it to non_clausal
        non_clausal.append(leaf)
        #Get the children of leaf and if there are any, append them to the list of leaves to explore
        children = tree[leaf]
        if len(children) > 0:
            leaves += children
    #return the amount of non_clausal members of the clause explored
    return len(non_clausal)

def findMeanLengthOfClause(conllu: pd.DataFrame, id_tree: dict[int, dict[int, list[int]]]):
    """
    Function that finds the mean length of clauses in a given conllu-snippet.
    Calculated by traversing the syntactic tree and seeing how many children start new clauses and how many belong to the head (obviosuly also including grandchildren etc.)
    """
    deprel_conllu = conllu['deprel']
    clauses = ['csubj', 'ccomp', 'xcomp', 'advcl', 'acl', 'acl:relcl', 'xcomp:ds', 'root']
    clause_lengths = []
    for head in id_tree:
        tree = id_tree[head]
        clausal_heads = []
        for i in tree:
            if deprel_conllu.iloc[i] in clauses:
                clausal_heads.append(i)
        #clausal_heads = [i for i in tree if deprel_conllu.iloc[i] in clauses]
        for ch in clausal_heads:
            clause_lengths.append(getNonClausalChildrenAmount(deprel_conllu, tree, ch))
    return np.mean(clause_lengths)

def getPOSVariation(conllu: pd.DataFrame, pos: str):
    """
    Function for getting the variation of words belonging to a sepcific POS category.
    Measures the ratio between unique words and all words, but in specific POS categories.
    """
    all_pos_tags_present = conllu['upos'].drop_duplicates().to_numpy(str)
    if pos not in all_pos_tags_present:
        return 0
    all_specific_pos = conllu[conllu['upos'] == pos]
    reduced_df = all_specific_pos[['text','upos']]
    uniq_words = reduced_df['text'].apply(lambda x: str(x).lower()).drop_duplicates()
    return len(uniq_words) / len(all_specific_pos)

def getCorrectedPOSVariation(conllu: pd.DataFrame, pos: str):
    """
    Same as POS variation, except we divide the amount of unique words by a 'corrected' term, rather than the raw number of words.
    The corrected term is equal to sqrt(2 * total number of words with specified POS tag)
    """
    all_pos_tags_present = conllu['upos'].drop_duplicates().to_numpy(str)
    if pos not in all_pos_tags_present:
        return 0
    all_specific_pos = conllu[conllu['upos'] == pos]
    reduced_df = all_specific_pos[['text','upos']]
    uniq_words = reduced_df['text'].apply(lambda x: str(x).lower()).drop_duplicates()
    return len(uniq_words) / np.sqrt(2*len(all_specific_pos))

def getRatioOfFunctionWords(conllu: pd.DataFrame):
    """
    Function for calculating the ratio between function words and content words.
    Essentially means dividing the number of function words (all other POS tags) by the number of content words (NOUN, PROPN, ADJ, NUM, and VERB)
    """
    num_of_content_words = len(conllu[(conllu['upos'] == 'NOUN') | (conllu['upos'] == 'PROPN') | (conllu['upos'] == 'ADJ') | (conllu['upos'] == 'NUM') | (conllu['upos'] == 'VERB')])
    #Don't divide by 0...
    if num_of_content_words == 0:
        return 1.0
    return (len(conllu)-num_of_content_words)/num_of_content_words

def getPreposingAdverbialClauses(corpus: dict[str,pd.DataFrame]) -> dict:
    """
    Function for calculating the number of preposing adverbial clauses in a conllu file
    """
    returnable = {}
    for key in corpus:
        df = corpus[key]
        #df['head'] = df['head'].apply(lambda x: int(x))
        #df['id'] = df['id'].apply(lambda x: int(x))
        prep_advcl = 0
        advcl_id = 1000
        root_id = 1000
        for i in range(len(df)):
            if df.loc[i]['id'] == '1':
                advcl_id = 1000
                root_id = 1000
            if df.loc[i]['deprel'] == 'root':
                root_id = i
            if df.loc[i]['deprel'] == 'advcl':
                advcl_id = i
                if advcl_id < root_id:
                    prep_advcl += 1
        returnable[key] = prep_advcl
    return returnable

