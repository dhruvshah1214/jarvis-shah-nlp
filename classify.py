from sklearn.externals import joblib
from nltk.tag import StanfordNERTagger
from nltk.tokenize import word_tokenize
import CONSTANTS
from train import featureExtraction
import argparse

def entityExtraction(inp):
    print CONSTANTS.NER_MODEL_PATH_FULL
    st = StanfordNERTagger(CONSTANTS.NER_MODEL_PATH_FULL, CONSTANTS.STANFORD_NER_JAR_PATH_FULL)
    return st.tag(word_tokenize(inp))

def classifyText(text): 
    classifier = joblib.load(CONSTANTS.MOD_MODEL_PATH_FULL)
    lbls = classifier.labels()
    moduleClass = classifier.classify(featureExtraction(text))
    entityExt = entityExtraction(text)
    return {'labels':lbls, 'module':moduleClass, 'entity': entityExt}
def main():
    parserN = argparse.ArgumentParser()
    parserN.add_argument('-s','--sentence', help='The command to classify', required=False)
    args = vars(parserN.parse_args())
    
    sent = args['sentence']
    
    if sent == None:
        sent = "Send Yeah dude to Hitesh"
    classifyText(sent)
    

if __name__ == "__main__":
    main()