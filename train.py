from nltk import word_tokenize
from sklearn.externals import joblib
import os.path
from nltk.corpus import stopwords
import CONSTANTS
import argparse
from nltk import FreqDist
from nltk import pos_tag
from nltk import NaiveBayesClassifier
from entityTrainer import call_Stanford_NER
from entityTrainer import ner_train

#helpers

def getCurrentTrainingData():
    if os.path.isfile(CONSTANTS.MOD_TRAIN_PATH_FULL):
        obj = joblib.load(CONSTANTS.MOD_TRAIN_PATH_FULL)
        return obj
    else:
        return []

def tagInput(inp):
    word_tag = word_tokenize(inp)
    pos_tag_sent = pos_tag(word_tag)
    return pos_tag_sent

def featureExtraction(sentence):

    # feature 1 -> tagged input
    features = {'taggedInput':tagInput(sentence), 'bow':{}}

    # bag of words

    # words not to include
    exclude_words = stopwords.words('english')
    for c in [".", "?", "!", ","]:
        exclude_words.append(c)
    arr_all_words = list(set([w for w in word_tokenize(sentence) if w not in exclude_words]))

    if os.path.isfile(CONSTANTS.BOW_PATH):
        for w in joblib.load(CONSTANTS.BOW_PATH):
            if w not in arr_all_words and w != None:
                arr_all_words.append(w)
        joblib.dump(arr_all_words, CONSTANTS.BOW_PATH)
    else:
        joblib.dump(arr_all_words, CONSTANTS.BOW_PATH)
    all_words = FreqDist(w.lower() for w in arr_all_words)
    word_features = all_words.keys()[:2000]
    document_words = set(word_tokenize(sentence))
    bow = {}
    for word in word_features:
        bow['contains(%s)' % word] = (word in document_words)
    features['bow'] = bow

    # add other features here... v
    return bow



#def entityExtraction(inp):
#    st = StanfordNERTagger('/Library/WebServer/Documents/JARVIS/nlp_module/python_app/JARVIS_NLP/ner-model-jarvis.ser.gz')
#    return st.tag(inp.split())



#program

def massTrainModule(sents, labels):
    #print featureExtraction(input)
    featureList = [featureExtraction(str(s)) for s in sents]
    labelList = [str(l) for l in labels]
    data = getCurrentTrainingData()

    newData = list(zip(featureList, labelList))

    for (a, b) in newData:
        data.append((a, b))

    joblib.dump(data, CONSTANTS.MOD_TRAIN_PATH_FULL)
    actuallyTrainClassifier()

def train(e, b, m, s, l):
    if e != None and bool(e) == True:
        if bool(b) != None:
            ner_train(bool(b), str(m))
    #print featureExtraction(input)
    data = getCurrentTrainingData()
    data.append((featureExtraction(str(s)), str(l)))

    joblib.dump(data, CONSTANTS.MOD_TRAIN_PATH_FULL)
    actuallyTrainClassifier()

def actuallyTrainClassifier():
    data = getCurrentTrainingData()
    classifier = NaiveBayesClassifier.train(data)
    joblib.dump(classifier, CONSTANTS.MOD_MODEL_PATH_FULL)

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1", "y")

def main():
    parserN = argparse.ArgumentParser()
    parserN.register('type','bool',str2bool) # add type keyword to registries
    parserN.add_argument('-e','--entity', help='Tagged or not', required=True, type='bool')
    parserN.add_argument('-i','--isbracket', help='is bracket or slash form', required=False, type='bool')
    parserN.add_argument('-s','--sentence', help='The sentence to train on', required=True)
    parserN.add_argument('-l','--label', help='The sentence\'s label', required=True)

    args = vars(parserN.parse_args())
    
    sentence_tokens = args['sentence'].split(' ')
    sentence_arr = [sent.split('/')[0] for sent in sentence_tokens]
    sent_natural = ' '.join(sentence_arr)
        
    train(args['entity'], args['isbracket'], args['sentence'], sent_natural, args['label'])

if __name__ == "__main__":
    main()


