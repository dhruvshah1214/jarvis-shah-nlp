import webapp2
from classify import entityExtraction
from classify import classifyText
import json
from train import train
from entityTrainer import ner_train

class Classify(webapp2.RequestHandler):

    def get(self):
        sent = self.request.GET['text']
        moduleC = classifyText(sent)
        self.response.headers['Content-Type'] = 'application/json'   
        resp = {
            'success': True, 
            'response': {'label':moduleC}
        } 
        self.response.write(json.dumps(resp))

class TrainModel(webapp2.RequestHandler):

    def get(self):
        ent_tagged = bool(self.request.GET['tagged'])
        bracket = bool(self.request.GET['bracket'])
        text = str(self.request.GET['text'])
        label = str(self.request.GET['label'])
        
        if ent_tagged == False:
            sentence_tokens = text.split(' ')
            sentence_arr = [sent.split('/')[0] for sent in sentence_tokens]
            sent_natural = ' '.join(sentence_arr)
        else:
            sent_natural = text
            
        train(ent_tagged, bracket, text, sent_natural, label)
        self.response.headers['Content-Type'] = 'application/json'   
        resp = {
            'success': True, 
            'response': None
        } 
        self.response.write(json.dumps(resp))
        
class TrainNERModel(webapp2.RequestHandler):

    def get(self):
        bracket = bool(self.request.GET['bracket'])
        text = str(self.request.GET['text'])
        
        ner_train(bracket, text)
        self.response.headers['Content-Type'] = 'application/json'   
        resp = {
            'success': True, 
            'response': None
        } 
        self.response.write(json.dumps(resp))
        
app = webapp2.WSGIApplication([
    ('/', Classify),
    ('/classify', Classify),
    ('/train', TrainModel),
    ('/trainner', TrainNERModel),
    ('/ner', TrainNERModel),
    ('/nertrain', TrainNERModel)
], debug=True)