import argparse
import ast
import CONSTANTS
import os
import subprocess

def call_Stanford_NER():
    origWD = os.getcwd() # remember our original working directory
    os.chdir(CONSTANTS.STANFORD_DIR_PATH_FULL)
    executeStr = ["java", "-cp",  "stanford-ner.jar:lib/*:.", "edu.stanford.nlp.ie.crf.CRFClassifier",  "-prop", CONSTANTS.NER_PROPERTY_FILE_PATH_FULL]
    subprocess.call(executeStr)
    os.chdir(origWD)

def ner_train(form, sentence):
    if str(form) == 'bracket' or bool(form) == True:
        sentence_arr = ast.literal_eval(sentence)
    else:
        sentence_tokens = sentence.split(' ')
        sentence_arr = [(sent.split('/')[0], sent.split('/')[1]) for sent in sentence_tokens]
    with open(CONSTANTS.NER_TRAINING_FILE_PATH_FULL, "a") as trainFile:
        for token, entity in sentence_arr:
            # print the token and entity to the next line of the .tsv training file
            line = token + '\t' + entity + '\n'
            trainFile.write(line)

    call_Stanford_NER()
    
def main():
    parser = argparse.ArgumentParser(description='Train the NER model.')

    parser.add_argument('-b','--bracket', help='The entity-tagged sentence', required=False)
    parser.add_argument('-s','--slash', help='The entity-tagged sentence', required=False)

    args = vars(parser.parse_args())


    if args['slash'] != None:
        sentence = args['slash']
        sentence_tokens = sentence.split(' ')
        sentence_arr = [(sent.split('/')[0], str.split('/')[1]) for sent in sentence_tokens]

    if args['bracket'] != None:
        sentence = args['bracket']
        sentence_arr = ast.literal_eval(sentence)

    with open(CONSTANTS.NER_TRAINING_FILE_PATH_FULL, "a") as trainFile:
        for token, entity in sentence_arr:
            # print the token and entity to the next line of the .tsv training file
            line = token + '\t' + entity + '\n'
            trainFile.write(line)

    call_Stanford_NER()

if __name__ == "__main__":
    main()

