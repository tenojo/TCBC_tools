#Imports
import pandas as pd
from TCBC_tools import Structure
from TCBC_tools import FeatureExtraction as fe
import json
import os
from datasets import Dataset

#Constants
#In the deprel column
#Dependency relation types in Finnish UD
DEPRELS = ['root', 'nsubj', 'advmod', 'obl', 'obj', 'conj', 'aux', 'cc', 'amod', 'nmod:poss', 'mark', 'cop', 'nsubj:cop', 'advcl', 'xcomp', 'case', 'det', 'ccomp', 'nmod', 'parataxis', 'acl:relcl', 'acl', 'xcomp:ds', 'discourse', 'nummod', 'fixed', 'cop:own', 'appos', 'flat:name', 'compound:nn', 'aux:pass', 'vocative', 'nmod:gobj', 'nmod:gsubj', 'compound:prt', 'csubj:cop', 'flat:foreign', 'orphan', 'cc:preconj', 'csubj', 'compound', 'flat', 'goeswith']

#In the feats column
#Cases in Finnihs UD
CASES = ['Case=Nom', 'Case=Gen', 'Case=Par', 'Case=Ill', 'Case=Ine', 'Case=Ela', 'Case=Ade', 'Case=All', 'Case=Ess', 'Case=Abl', 'Case=Tra', 'Case=Acc', 'Case=Ins', 'Case=Abe', 'Case=Com']
#Verb forms in Finnish UD
VERBFORMS = ['VerbForm=Fin', 'VerbForm=Inf', 'VerbForm=Part']
#Verb tenses in Finnish UD
VERBTENSES = ['Tense=Pres', 'Tense=Past']
#Verb voices in Finnish UD
VERBVOICES = ['Voice=Act', 'Voice=Pass']
#Verb moods in Finnish UD
VERBMOODS = ['Mood=Ind', 'Mood=Cnd', 'Mood=Imp']
#Verb 'person' in Finnish UD (aka first person, second person and so on)
PERSONS = ['Person=0', 'Person=1', 'Person=2', 'Person=3']
#Verb 'number' in Finnish UD (aka first singluar person [me] or first plural person [we] and so on)
NUMBERS = ['Number=Sing', 'Number=Plur']
#Connegative (aka verb that has been given a negative meaning by 'ei')
CONNEGATIVE = ['Connegative=Yes']
#Degrees in Finnish UD (positive, comparative, and superlative)
DEGREES = ['Degree=Pos','Degree=Cmp','Degree=Sup']
#Syles in Finnish UD
STYLES = ['Style=Arch', 'Style=Coll']
#Reflex pronouns in Finnish UD
REFS = ['Reflex=Yes']
#PronTypes in Finnish UD
PRONTYPES = ['PronType=Dem', 'PronType=Ind', 'PronType=Int', 'PronType=Prs', 'PronType=Rcp', 'PronType=Rel']
#Verb polarity in Finnish UD
POLARITY = ['Polarity=Neg']
#Person possessor in Finnish UD (e.g. luu VS. luumme)
PPSORS = ['Person[psor]=1', 'Person[psor]=2', 'Person[psor]=3']
#Partforms in Finnish UD
PARTFORMS = ['PartForm=Agt', 'PartForm=Neg', 'PartForm=Past', 'PartForm=Pres']
#Number types in Finnish UD
NUMTYPES = ['NumType=Card', 'NumType=Ord']
#Numeral posessor in Finnish UD (e.g. aikani VS. aikanamme)
NPSORS = ['Number[psor]=Plur', 'Number[psor]=Sing']
#Infinitive forms for verbs in Finnish UD
INFFORMS = ['InfForm=1', 'InfForm=2', 'InfForm=3']
#Marks foreign words in Finnish UD
FOREIGN = ['Foreign=Yes']
#Derivations of words in Finnish UD
DERIVATIONS = ['Derivation=Inen', 'Derivation=Ja', 'Derivation=Lainen', 'Derivation=Llinen', 'Derivation=Tar', 'Derivation=Ton', 'Derivation=Ttain', 'Derivation=U', 'Derivation=Vs', 'Derivation=Inen|Vs' 'Derivation=Ja|Tar', 'Derivation=Lainen|Vs', 'Derivation=Llinen|Vs', 'Derivation=Ton|Vs']
#Clitics of words in Finnish UD
CLITICS = ['Cilitic=Han', 'Cilitic=Ka', 'Cilitic=Kaan', 'Cilitic=Kin', 'Cilitic=Ko', 'Cilitic=Pa', 'Cilitic=S', 'Cilitic=Han|Kin', 'Cilitic=Han|Ko', 'Cilitic=Han|Pa', 'Cilitic=Ko|S', 'Cilitic=Pa|S']
#AdpTypes in Finnish UD
ADPTYPES = ['AdpType=Post', 'AdpType=Prep']
#Marks if words are abbrevations in Finnish UD
ABBR = ['Abbr=Yes']

FEATS = CASES + VERBFORMS + VERBTENSES + VERBVOICES + VERBMOODS + PERSONS + NUMBERS + CONNEGATIVE + DEGREES + STYLES + REFS + PRONTYPES + POLARITY + PPSORS + PARTFORMS + NUMTYPES + NPSORS + INFFORMS + FOREIGN + DERIVATIONS + CLITICS + ADPTYPES + ABBR
#In the upos column
#POS tags in Finnish UD
POS = ['NOUN', 'VERB', 'PRON', 'ADV', 'AUX', 'ADJ', 'PROPN', 'CCONJ', 'SCONJ', 'ADP', 'NUM', 'INTJ', 'PUNCT']

#Bigrams and trigrams
POS_BIGRAMS = []
POS_TRIGRAMS = []
for x in POS:
    for y in POS:
            for z in POS:
                  POS_TRIGRAMS.append((x,y,z))
            POS_BIGRAMS.append((x,y))

#Can be used if we want to look at all possible deprel bigrams and trigrams (not too computationally expensive, but a bit slow and can cause memory issues for a normal working laptop)
"""
DEPREL_BIGRAMS = []
DEPREL_TRIGRAMS = []
for x in DEPRELS:
    for y in DEPRELS:
            for z in DEPRELS:
                  DEPREL_TRIGRAMS.append((x,y,z))
            DEPREL_BIGRAMS.append((x,y))
"""

#NOT IN TCBC 1.0
#Uneccessary features and bi/trigrams, which don't occur even once in any of the books
#We remove them to save time when creating feature vectors
#THIS SHOULD BE UPDATED WITH NEW VERSIONS OF THE CORPUS

FEATS_TO_DEL = ['Derivation=Inen|VsDerivation=Ja|Tar', 'Derivation=Lainen|Vs', 'Derivation=Llinen|Vs', 'Derivation=Ton|Vs', 'Cilitic=Han', 'Cilitic=Ka', 'Cilitic=Kaan', 'Cilitic=Kin', 'Cilitic=Ko', 'Cilitic=Pa', 'Cilitic=S', 'Cilitic=Han|Kin', 'Cilitic=Han|Ko', 'Cilitic=Han|Pa', 'Cilitic=Ko|S', 'Cilitic=Pa|S']
POS_TRIGRAMS_TO_DEL = [('NOUN', 'INTJ', 'ADP'), ('VERB', 'ADP', 'SCONJ'), ('VERB', 'ADP', 'INTJ'), ('VERB', 'NUM', 'INTJ'), ('VERB', 'INTJ', 'ADP'), ('PRON', 'INTJ', 'CCONJ'), ('PRON', 'INTJ', 'NUM'), ('ADV', 'NUM', 'INTJ'), ('ADV', 'INTJ', 'ADP'), ('AUX', 'CCONJ', 'ADP'), ('AUX', 'SCONJ', 'CCONJ'), ('AUX', 'SCONJ', 'ADP'), ('AUX', 'SCONJ', 'INTJ'), ('AUX', 'ADP', 'INTJ'), ('AUX', 'INTJ', 'CCONJ'), ('AUX', 'INTJ', 'ADP'), ('ADJ', 'PRON', 'INTJ'), ('ADJ', 'SCONJ', 'CCONJ'), ('ADJ', 'ADP', 'INTJ'), ('ADJ', 'NUM', 'INTJ'), ('ADJ', 'INTJ', 'PRON'), ('ADJ', 'INTJ', 'AUX'), ('ADJ', 'INTJ', 'ADP'), ('ADJ', 'INTJ', 'NUM'), ('PROPN', 'ADJ', 'INTJ'), ('PROPN', 'SCONJ', 'CCONJ'), ('PROPN', 'SCONJ', 'ADP'), ('PROPN', 'SCONJ', 'INTJ'), ('PROPN', 'ADP', 'INTJ'), ('PROPN', 'INTJ', 'CCONJ'), ('PROPN', 'INTJ', 'SCONJ'), ('PROPN', 'INTJ', 'ADP'), ('PROPN', 'INTJ', 'NUM'), ('CCONJ', 'CCONJ', 'ADP'), ('CCONJ', 'CCONJ', 'NUM'), ('CCONJ', 'ADP', 'SCONJ'), ('CCONJ', 'ADP', 'INTJ'), ('CCONJ', 'NUM', 'SCONJ'), ('CCONJ', 'NUM', 'INTJ'), ('CCONJ', 'INTJ', 'ADP'), ('SCONJ', 'PROPN', 'INTJ'), ('SCONJ', 'CCONJ', 'AUX'), ('SCONJ', 'CCONJ', 'ADP'), ('SCONJ', 'CCONJ', 'NUM'), ('SCONJ', 'CCONJ', 'INTJ'), ('SCONJ', 'CCONJ', 'PUNCT'), ('SCONJ', 'SCONJ', 'ADP'), ('SCONJ', 'SCONJ', 'INTJ'), ('SCONJ', 'ADP', 'AUX'), ('SCONJ', 'ADP', 'CCONJ'), ('SCONJ', 'ADP', 'SCONJ'), ('SCONJ', 'ADP', 'ADP'), ('SCONJ', 'ADP', 'INTJ'), ('SCONJ', 'ADP', 'PUNCT'), ('SCONJ', 'NUM', 'SCONJ'), ('SCONJ', 'NUM', 'INTJ'), ('SCONJ', 'INTJ', 'ADJ'), ('SCONJ', 'INTJ', 'CCONJ'), ('SCONJ', 'INTJ', 'ADP'), ('SCONJ', 'INTJ', 'NUM'), ('ADP', 'VERB', 'INTJ'), ('ADP', 'ADV', 'INTJ'), ('ADP', 'ADJ', 'INTJ'), ('ADP', 'PROPN', 'INTJ'), ('ADP', 'SCONJ', 'CCONJ'), ('ADP', 'ADP', 'SCONJ'), ('ADP', 'ADP', 'INTJ'), ('ADP', 'NUM', 'SCONJ'), ('ADP', 'NUM', 'INTJ'), ('ADP', 'INTJ', 'PRON'), ('ADP', 'INTJ', 'ADV'), ('ADP', 'INTJ', 'AUX'), ('ADP', 'INTJ', 'ADJ'), ('ADP', 'INTJ', 'CCONJ'), ('ADP', 'INTJ', 'SCONJ'), ('ADP', 'INTJ', 'ADP'), ('ADP', 'INTJ', 'NUM'), ('NUM', 'PRON', 'INTJ'), ('NUM', 'ADV', 'INTJ'), ('NUM', 'AUX', 'CCONJ'), ('NUM', 'AUX', 'INTJ'), ('NUM', 'ADJ', 'INTJ'), ('NUM', 'CCONJ', 'ADP'), ('NUM', 'SCONJ', 'ADV'), ('NUM', 'SCONJ', 'CCONJ'), ('NUM', 'SCONJ', 'SCONJ'), ('NUM', 'SCONJ', 'ADP'), ('NUM', 'SCONJ', 'INTJ'), ('NUM', 'SCONJ', 'PUNCT'), ('NUM', 'ADP', 'ADP'), ('NUM', 'ADP', 'INTJ'), ('NUM', 'INTJ', 'VERB'), ('NUM', 'INTJ', 'AUX'), ('NUM', 'INTJ', 'ADP'), ('NUM', 'INTJ', 'INTJ'), ('INTJ', 'VERB', 'ADP'), ('INTJ', 'ADV', 'ADP'), ('INTJ', 'AUX', 'CCONJ'), ('INTJ', 'AUX', 'SCONJ'), ('INTJ', 'AUX', 'ADP'), ('INTJ', 'AUX', 'NUM'), ('INTJ', 'ADJ', 'ADP'), ('INTJ', 'ADJ', 'NUM'), ('INTJ', 'PROPN', 'SCONJ'), ('INTJ', 'CCONJ', 'CCONJ'), ('INTJ', 'CCONJ', 'ADP'), ('INTJ', 'CCONJ', 'NUM'), ('INTJ', 'SCONJ', 'CCONJ'), ('INTJ', 'SCONJ', 'ADP'), ('INTJ', 'SCONJ', 'NUM'), ('INTJ', 'SCONJ', 'INTJ'), ('INTJ', 'ADP', 'VERB'), ('INTJ', 'ADP', 'AUX'), ('INTJ', 'ADP', 'ADJ'), ('INTJ', 'ADP', 'CCONJ'), ('INTJ', 'ADP', 'SCONJ'), ('INTJ', 'ADP', 'ADP'), ('INTJ', 'ADP', 'NUM'), ('INTJ', 'ADP', 'INTJ'), ('INTJ', 'NUM', 'PRON'), ('INTJ', 'NUM', 'AUX'), ('INTJ', 'NUM', 'PROPN'), ('INTJ', 'NUM', 'CCONJ'), ('INTJ', 'NUM', 'SCONJ'), ('INTJ', 'NUM', 'ADP'), ('INTJ', 'NUM', 'NUM'), ('INTJ', 'NUM', 'INTJ'), ('INTJ', 'INTJ', 'ADP'), ('PUNCT', 'ADP', 'SCONJ'), ('PUNCT', 'ADP', 'INTJ')]

FEATS = [x for x in FEATS if x not in FEATS_TO_DEL]
POS_TRIGRAMS = [x for x in POS_TRIGRAMS if x not in POS_TRIGRAMS_TO_DEL]

#These lists can be generated by generating feature vectors for all books in TCBC and running the following code on the feature vectors:
"""
def zeroedFeatures(arr):
    zeroed_indices = []
    res = np.count_nonzero(arr, axis=0)
    for i in index_dict:
        if res[i] == 0:
            zeroed_indices.append(i)
    return zeroed_indices

to_del = zeroedFeatures(np.array([x[1] for x in results]))

def ngramToTuple(ngram_str):
    return tuple(ngram_str.split('_'))

to_del_feats = []
to_del_pos_bigram = []
to_del_pos_trigram = []
to_del_deprel_bigram = []
to_del_deprel_trigram = []
for i in to_del:
    if index_dict[i] in FEATS:
        to_del_feats.append(index_dict[i])
    test = ngramToTuple(index_dict[i])
    if test in POS_BIGRAMS:
        to_del_pos_bigram.append(test)
    if test in POS_TRIGRAMS:
        to_del_pos_trigram.append(test)
    if test in DEPREL_BIGRAMS:
        to_del_deprel_bigram.append(test)
    if test in DEPREL_TRIGRAMS:
        to_del_deprel_trigram.append(test)
"""

#Combine UD features together
CONLLU_FEATS = DEPRELS + FEATS + POS 


# Functions

def getKeylist(path_name):
    keylists = []
    with open(path_name, 'r') as f:
        for line in f:
            keylists.append(json.loads(line))
    return keylists

def minMaxNormalization(min_vector: list, max_vector:list, feature_vector:list):
    """
    Helper function for performing min-max normalization for feature vectors
    """
    to_return = []
    for i in range(len(feature_vector)):
        min_max_neg = (max_vector[i]-min_vector[i])
        if min_max_neg == 0:
            to_return.append(0)
        else:
            to_return.append((feature_vector[i]-min_vector[i])/(max_vector[i]-min_vector[i]))
    return to_return

def scaleCorpusData(corpus_data: dict[str,float], scaling_data: dict[str,float]):
    """
    Function for scaling some previously calculated data (e.g. word counts) with some other measure (e.g. number of sentences)
    """
    returnable = {}
    for key in corpus_data:
        returnable[key] = corpus_data[key]/scaling_data[key]
    return returnable

def buildDatasetFromRawConllus(conllu_folder):
    to_convert = []
    for key in os.listdir(conllu_folder):
        with open(conllu_folder+key, 'r', encoding='UTF-8') as reader:
            conllu_text = reader.read()
        to_convert.append({'book_id':key, 'data':conllu_text})
    return Dataset.from_list(to_convert)

def customConlluVectorizer(df: pd.DataFrame, generate_key_dictionary:bool=False):
    """
    Custom vecotrizer used to create feature vectors from hand-picked features.
    If passed a flag, will generate a dictionary that maps feature vector indices to the names of the features.
    Recommend only generating to key dictionary once to save some time, allthough it is not computationally super expensive.
    """
    feature_vector = []
    if generate_key_dictionary:
        feature_indices = {}
        index = 0
    temp_corp = {'1':df}
    syntactic_tree = Structure.buildIdTreeFromConllu(df)
    word_freqs = fe.getColumnFrequencies(temp_corp, ['text'])
    word_amounts = fe.getTokenAmounts(temp_corp)
    sent_amounts = fe.getNumOfSentences(temp_corp)

    #Deprel per main clause for each deprel
    for deprel in DEPRELS:
        feature_vector.append(fe.getDeprelFeaturePerBook(temp_corp, deprel, True)['1'])
        if generate_key_dictionary:
            feature_indices[index] = deprel
            index += 1
    #Feat per main clause for each feature
    for feat in FEATS:
        feature_vector.append(fe.getFeatsFeaturePerBook(temp_corp, feat, True)['1'])
        if generate_key_dictionary:
            feature_indices[index] = feat
            index += 1
    #POS related (simple) features
    for pos in POS:
        #POS per main clause for each pos-tag
        feature_vector.append(fe.getPosFeaturePerBook(temp_corp, pos, True)['1'])
        if generate_key_dictionary:
            feature_indices[index] = pos
            index += 1
        #POS pharses per main clause
        feature_vector.append(scaleCorpusData(fe.getPosPhraseCounts(temp_corp, pos), sent_amounts)['1'])
        if generate_key_dictionary:
            feature_indices[index] = pos+"_Phrase"
            index += 1
        #POS variation
        feature_vector.append(fe.getPOSVariation(df, pos))
        if generate_key_dictionary:
            feature_indices[index] = pos+"_Variation"
            index += 1
        #Corrected POS variation
        feature_vector.append(fe.getCorrectedPOSVariation(df, pos))
        if generate_key_dictionary:
            feature_indices[index] = pos+"_Variation_Corrected"
            index += 1
        #POS ratios
        pos2 = POS.copy()
        pos2.remove(pos)
        for pos_2 in pos2:
            #Check that we don't divide by 0 by accident!
            divider = fe.getPosFeaturePerBook(temp_corp, pos_2)['1']
            if divider == 0:
                feature_vector.append(0)
            else:
                feature_vector.append(fe.getPosFeaturePerBook(temp_corp, pos)['1'] / divider)
            if generate_key_dictionary:
                feature_indices[index] = pos+"_To_"+pos_2+"_Ratio"
                index += 1
    """
    #Flat POS bigrams per main clause
    pos_bigrams = bdf.getPosNGramForCorpus(temp_corp, 2)['1']
    for pb in POS_BIGRAMS:
         feature_vector.append(pos_bigrams.get(pb, 0) / sent_amounts['1'])
         if generate_key_dictionary:
              feature_indices[index] = 'flat_'+pb[0]+'_'+pb[1]
              index += 1
    #Flat POS trigrams per main clause
    pos_trigrams = bdf.getPosNGramForCorpus(temp_corp, 3)['1']
    for pt in POS_TRIGRAMS:
         feature_vector.append(pos_trigrams.get(pt, 0) / sent_amounts['1'])
         if generate_key_dictionary:
              feature_indices[index] = 'flat_'+pt[0]+'_'+pt[1]+'_'+pt[2]
              index += 1
    #Tree POS bigrams per main clause
    pos_bigrams = bdf.getSyntacticTreeNGram(df, syntactic_tree, 2, 'upos')
    for pb in POS_BIGRAMS:
         feature_vector.append(pos_bigrams.get(pb, 0) / sent_amounts['1'])
         if generate_key_dictionary:
              feature_indices[index] = 'tree_'+pb[0]+'_'+pb[1]
              index += 1
    #Tree POS trigrams per main clause
    pos_trigrams = bdf.getSyntacticTreeNGram(df, syntactic_tree, 3, 'upos')
    for pt in POS_TRIGRAMS:
         feature_vector.append(pos_trigrams.get(pt, 0) / sent_amounts['1'])
         if generate_key_dictionary:
              feature_indices[index] = 'tree_'+pt[0]+'_'+pt[1]+'_'+pt[2]
              index += 1
    """

        

    #Other features
    #TTR
    feature_vector.append(fe.getTypeTokenRatios(word_freqs, word_amounts)['1'])
    if generate_key_dictionary:
            feature_indices[index] = "TTR"
            index += 1
    #MLS
    feature_vector.append(fe.getAvgSentenceLens(temp_corp)['1'])
    if generate_key_dictionary:
            feature_indices[index] = "MLS"
            index += 1
    #Average number of syllables per sentence
    feature_vector.append(fe.getAverageSyllablesPerSentence(df, syntactic_tree))
    if generate_key_dictionary:
            feature_indices[index] = "AvgSylPerSent"
            index += 1
    #CONJ2Sent
    feature_vector.append(fe.getConjPerSentence(temp_corp)['1'])
    if generate_key_dictionary:
            feature_indices[index] = "ConjPerSent"
            index += 1
    #Flesch-Kincaid grade level
    feature_vector.append(fe.getFleschKincaidGradeLevel(temp_corp)['1'])
    if generate_key_dictionary:
            feature_indices[index] = "F-K-GradeLevel"
            index += 1
    #Get modified SMOG Index
    feature_vector.append(fe.getModifiedSmogIndexForFinnish(df, syntactic_tree))
    if generate_key_dictionary:
            feature_indices[index] = "SMOG"
            index += 1
    #Get Coleman-Liau index
    feature_vector.append(fe.getColemanLiauIndex(df, syntactic_tree))
    if generate_key_dictionary:
            feature_indices[index] = "ColemanLiau"
            index += 1
    #Get Automated readability index
    feature_vector.append(fe.getARI(df, syntactic_tree))
    if generate_key_dictionary:
            feature_indices[index] = "ARI"
            index += 1

    #Preposing adverbial clauses
    feature_vector.append(fe.getPreposingAdverbialClauses(temp_corp)['1'])
    if generate_key_dictionary:
            feature_indices[index] = "PrepAdvcl"
            index += 1
    #Ratio between function words and content words
    feature_vector.append(fe.getRatioOfFunctionWords(df))
    if generate_key_dictionary:
            feature_indices[index] = "Func2ContWordRatio"
            index += 1
    #Features that require parsing the syntactic tree structure
    #Average depth of syntactic tree
    feature_vector.append(fe.getMeanSyntacticTreeDepth(syntactic_tree))
    if generate_key_dictionary:
            feature_indices[index] = "MeanTreeDepth"
            index += 1
    #Maximum depth of syntactic tree
    feature_vector.append(fe.getMaxSyntacticTreeDepth(syntactic_tree))
    if generate_key_dictionary:
            feature_indices[index] = "MaxTreeDepth"
            index += 1
    #Nesting of clauses
    feature_vector.append(len(fe.findNestingSubclauses(df, syntactic_tree)))
    if generate_key_dictionary:
            feature_indices[index] = "NestingOfClauses"
            index += 1
    #Stacking of clauses
    feature_vector.append(len(fe.findStackingClauses(df, syntactic_tree)))
    if generate_key_dictionary:
            feature_indices[index] = "StackingOfClauses"
            index += 1
    #Mean length of clauses
    feature_vector.append(fe.findMeanLengthOfClause(df, syntactic_tree))
    if generate_key_dictionary:
            feature_indices[index] = "MLC"
            index += 1
    #deprel bigrams per main clause
    """
    deprel_bigrams = bdf.getSyntacticTreeNGram(df, syntactic_tree, 2)
    for db in DEPREL_BIGRAMS:
         feature_vector.append(deprel_bigrams.get(db, 0) / sent_amounts['1'])
         if generate_key_dictionary:
              feature_indices[index] = db[0]+'_'+db[1]
              index += 1
    """
    #deprel trigrams per main clause
    """
    deprel_trigrams = bdf.getSyntacticTreeNGram(df, syntactic_tree, 3)
    for dt in DEPREL_TRIGRAMS:
         feature_vector.append(deprel_trigrams.get(dt, 0) / sent_amounts['1'])
         if generate_key_dictionary:
              feature_indices[index] = dt[0]+'_'+dt[1]+'_'+dt[2]
              index += 1
    """
    


    if generate_key_dictionary:
         return feature_vector, feature_indices
    return feature_vector