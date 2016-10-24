import json
import train
import CONSTANTS
from nltk import word_tokenize

file = CONSTANTS.WORKING_DIR_PATH_FULL + 'training_data/LUISAPP.json'


with open(file) as jsonFile:
    JSON = json.load(jsonFile)


texts = []
labels = []

for obj in JSON:
    t = obj["text"]
    i = obj["intent"]
    entities = obj["entities"]
    markUp = word_tokenize(t)
    for e in entities:
        entityName = e["entity"]
        spos = e["startPos"]
        epos = e["endPos"]
        for ind in range(spos, epos):
            markUp[ind] = markUp[ind] + '/' + entityName.upper()
    texts.append(t)
    labels.append(i)
train.massTrainModule(texts, labels)
print texts
print labels
