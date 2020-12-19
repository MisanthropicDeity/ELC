import torch
import pandas as pd
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

df = pd.read_csv('data/faqs/faq_covidbert.csv')

ques = df["question"][2]
sentences = df['answer'].head(5)
text = " ''' "
for a in sentences:
    #text+='<p>'
    text+=a
    #text+='<\p>'
text += " ''' "
print (ques)


model_name = "deepset/roberta-base-squad2-covid"
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name, args_parser= [])
QA_input = {
    'question': ques,
    'context': text,
}


res = nlp(QA_input,topk = 2)

print(res)