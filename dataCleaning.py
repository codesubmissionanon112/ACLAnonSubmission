import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
#https://github.com/JULIELab/EmoBank/tree/master/corpus
#https://saifmohammad.com/WebPages/nrc-vad.html
def init_emo(): 
    data = pd.read_csv('data/EmoBank-master/corpus/emobank.csv')
    #data starts as 1 to 5, for VAD so normalising it
    data['V'] = data['V'].apply(lambda x : x/5)
    data['A'] = data['A'].apply(lambda x : x/5)
    data['D'] = data['D'].apply(lambda x : x/5)

    data = data.drop(['id'], axis=1)
    test  = data[data['split'] == 'test']
    test = test.drop(['split'], axis=1)
    train  = data[data['split'] == 'train']
    train = train.drop(['split'], axis=1)

    dataLexicon = pd.read_csv('data/NRC-VAD-Lexicon/NRC-VAD-Lexicon.txt',  names=['text', 'V', 'A', 'D'], header=None, sep = '\t')
    trainLex, testLex = train_test_split(dataLexicon, test_size=0.3, random_state=42)

    fullTrain = pd.concat([train, trainLex], ignore_index = True)
    fullTest = pd.concat([test, testLex], ignore_index = True)
    #print(fullTrain)
    #print(fullTest)

    return fullTest, fullTrain#, valid
