# def imports():   
import streamlit as st
import pandas as pd
import numpy as np
import time
import json
import torch
import pandas as pd
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline


@st.cache
def pr_task3():
    f = open('Covid.json', encoding="UTF-8")
    data1 = json.load(f)
    table = pd.DataFrame.from_dict(data1)
    table = table.head(80)
    return table
    
def non_cachable_task3(): 
    model_name2 = 'google/tapas-medium-finetuned-wikisql-supervised'
    table_ans = pipeline('table-question-answering', model = model_name2, tokenizer = model_name2)
    return table_ans


def non_cachable_task1():
    #global nlp1
    model_name3 = 't5-small'
    nlp1 = pipeline('summarization', model = model_name3)
    return nlp1

@st.cache
def pr_task2():
    df = pd.read_csv('data/faqs/faq_covidbert.csv')
    # ques = df["question"][2]
    sentences = df['answer']
    text_task2 = " ''' "
    for a in sentences:
        #text+='<p>'
        text_task2 +=a
        #text+='<\p>'
    text_task2 += " ''' "
    return text_task2

def non_cachable_task2():
    model_name = "deepset/roberta-base-squad2-covid"
    nlp = pipeline('question-answering', model=model_name, tokenizer=model_name, args_parser= [])
    return nlp


st.title('ELC PROJECT')

choice = ['Summary', 'Question Answer (Theoretical)', 'Question Answer (Statistical)' ]


option = st.sidebar.selectbox(
    'Select an NLP Task from the given list',
     choice,
     )


if(option == 'Summary'):
    #initializations()
    st.write('## Summary ')
    user_input = st.text_area("Content to summarize", 'Enter your text here')
    values = st.sidebar.slider('Select a range of values', 10, 100, (25, 75))
    st.sidebar.write('Min:', values[0])
    st.sidebar.write('Max:', values[1])
    if st.button('Compute'):
        nlp1 = non_cachable_task1()
        summarized_text = nlp1(user_input,min_length = values[0],max_length = values[1])
        st.write(summarized_text[0]['summary_text'])
        st.write("Length of summary: " ,len(summarized_text[0]['summary_text'].split()) )

elif  option == 'Question Answer (Theoretical)' :
    st.write('## Question Answer Theoretical ')
    user_input = st.text_input("Ask a question", 'Enter question here')
    value = st.sidebar.slider('Select number of closest answer to ',1,5, (1))
    st.sidebar.write("No of Answers to Print", value)
    text_ans = pr_task2()
    if st.button('compute'):
        nlp = non_cachable_task2()
        QA_input = {
        'question': user_input,
        'context': text_ans,
        }
        res = nlp(QA_input,topk = value)
        #st.write(res)
        if(value>1):
            for x in range(len(res)):
                st.write('Answer ',x+1," :", res[x]['answer'])
                st.write('score : ', res[x]['score'] )
        else:
            st.write('Answer ',1," :", res['answer'])
            st.write('score : ', res['score'] )

    
        
elif  option == 'Question Answer (Statistical)' :
    st.write('## Question Answer Statistical ')
    user_input = st.text_input("Ask a question", 'Enter question here')
    # value = st.sidebar.slider('Select number of closest answer to ',1,5, (1))
    # st.sidebar.write("No of Answers to Print", value)
    table = pr_task3()
    sh = 0
    if st.checkbox('Show data'):
        st.write(table)
    if st.button('compute'):
        nlp = non_cachable_task3()
        res = nlp(table,user_input)
        #st.write(res)
        if len(res['answer'])>0:
            st.write('Answer :', res['answer'])
        else:
            st.write("Sorry our model wasn't able to give answer to your question this time, try some other query please")


# latest_iteration = st.empty()
# bar = st.progress(0)

# for i in range(100):
#   # Update the progress bar with each iteration.
#   latest_iteration.text(f'Iteration {i+1}')
#   bar.progress(i + 1)
#   time.sleep(0)
