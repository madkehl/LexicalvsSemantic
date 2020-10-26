##borrowed from https://github.com/plotly/dash-sample-apps/tree/master/apps/dash-clinical-analytics
#key imports
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, ClientsideFunction, State
import dash_bootstrap_components as dbc
import base64
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
import math

import collections
from collections import Counter

import nltk
from nltk import ConcordanceIndex

import spacy
nlp = spacy.load("en_core_web_md")
vocab = nlp.vocab.strings


from string import punctuation
from re import sub
punctuation = punctuation +'”'+'“'+'’' + '—' + '’' + '‘' +'0123456789'

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

stopwords = ['whom','hast','thou','therein', 'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now']

test_doc = open("./test_doc.txt", "r", encoding = "utf-8")
test_doc = test_doc.read()


# external CSS stylesheets
external_stylesheets = [
   dbc.themes.YETI
]


#config settings
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}])

server = app.server
app.config.suppress_callback_exceptions = True

def most_similar(word, topn=10):
    #https://stackoverflow.com/questions/57697374/list-most-similar-words-in-spacy-in-pretrained-model
    ms = nlp.vocab.vectors.most_similar(nlp(word).vector.reshape(1,nlp(word).vector.shape[0]), n=topn)
    words = [nlp.vocab.strings[w] for w in ms[0][0]]
    distances = ms[2]
    return words, distances


def latent_meaning_spacy(i, top_ = 10):
    '''
    INPUT: word tuple, topn
    
    OUTPUT: word tuple, the distance between the two original words, and the distance between the topn related words
    '''
    if(i[0] in vocab) & (i[1] in vocab):
        
        first_close, first_close_distances = most_similar(i[0], topn= top_)
        second_close, second_close_distances = most_similar(i[1], topn= top_)
        first_vec = nlp.vocab[i[0]].vector
        second_vec = nlp.vocab[i[1]].vector
        item_dis = np.dot(first_vec, second_vec)/(np.linalg.norm(first_vec)*np.linalg.norm(second_vec))
        
        for z in first_close:
            first_vec = first_vec + nlp.vocab[z].vector
        
        for z in second_close:
            second_vec = second_vec +  nlp.vocab[z].vector
        
        first_vec = first_vec - nlp.vocab[i[0]].vector
        second_vec = second_vec - nlp.vocab[i[1]].vector
        
        latent_dis = np.dot(first_vec, second_vec)/(np.linalg.norm(first_vec)*np.linalg.norm(second_vec))
        
        return([i, item_dis, latent_dis])
    else:
        return([None, None, None])

def make_ci(komyagin):
    txt = (nltk.Text(komyagin))
    return(ConcordanceIndex(txt))

#for some reason having difficulty subsetting by tokens instead of characters:: instead just use enough 
#characters to later be able to reliably get a 5 word radius

def get_context(ci, word, width=150, lines=100):
    
    half_width =  (width - len(word) - 2) // 2
    context = width // 4 # approx number of words of context
    num = 5
    results = []
    offsets = ci.offsets(word)
    
    if offsets:
        for i in offsets:
            query_word = ci._tokens[i]
  
            left_context = ci._tokens[max(0, i - context) : i]
            right_context = ci._tokens[i + 1 : i + context]
           
            left_print = " ".join(left_context)[-half_width:]
            right_print = " ".join(right_context)[:half_width]
                
            full_line_print = " ".join([left_print, query_word, right_print])
            
            results.append(full_line_print)
            
    return ([num, results])

def clean_text(txt_ls):
    
    translator = str.maketrans('','', sub('\#', '', punctuation))

    clean_txt_ls = []
    for i in txt_ls:
        n = i.split()
        str_ = ""
        for z in n:
            z = z.lower()
            s = z.translate(str.maketrans(translator))
            if s not in stopwords:
          #  print(s)
                str_ = str_ + " " + s
        clean_txt_ls.append(str_[1:])
        
    return(clean_txt_ls)

def clean_context(ci, target_word1, target_word2, window = 5):
    tox = []
    #gets context around a target word
    word_one_context = get_context(ci, target_word1)
   
     #takes this context and reshapes it into a -5 to + 5 window
    for i in word_one_context[1]:
        split_i = i.split()
        for z in split_i:
            tox.append(z)
            
    to_mend = list(enumerate(tox))
    
    mended_tox = []
    #modifications necessary to account for words at the start and end of corpus
    for z in to_mend:
        if z[1] == target_word1:
            if z[0] < (window):
                padded = tox[0:(z[0]+window)]
                mended_tox.append(padded)
            elif z[0] > (len(tox) - 1):
                padded = tox[(z[0] -window):(len(tox)-1)]
                mended_tox.append(padded)
            else: 
                padded = tox[(z[0] -window):(z[0] + window)]
                mended_tox.append(padded)
#reshaping list into format that can be more quickly processed    
    final_tox = []
    for i in mended_tox:
        for n in i:
            final_tox.append(n)
    return(final_tox)

#gathers all contexts of the word in results
def mutual_informativity(ci, target_word1, target_word2, total_count, window = 10):
   
    final_tox =  clean_context(ci, target_word1, target_word2, window = 10)
    
#this will return the count of target_word2 in the vicinity of target_word1
    prob_numx = Counter(final_tox)
    prob_num = prob_numx[target_word2]
    P_rc = prob_num/total_count
    P_r = (count(ci, target_word1))/total_count
    P_c = (count(ci, target_word2))/total_count
 #checking for indexing errors and overlap errors.
#fix to overlap is a bit of a hack fix, most notable problems are double counting 
#so if number of cooccurances is greater than number of times a word appears in 
#a document, then it replaces cooccurance with the total number of times
#except in the case that this is an odd number (since it's not being double-counted then,
#it's being double counted once and has another appearance)
#odd numbers, is P_c - 1/total. 
#this still may cause some errors
    if P_rc == 0:
     #   print(prob_numx)
        return ([target_word1, target_word2, "ERROR"])
    elif P_rc > P_c:
     #   print(target_word2)
        rounded = int(P_rc/P_c)
        nr = P_rc/P_c
        if (rounded > nr):
            P_n = P_c - 1/total_count
        else:
            P_n = P_c
        mutinf = math.log10(P_n/(P_r*P_c))
        #print([P_rc, P_c, P_r, P_n])
    elif P_rc > P_r:
        rounded = int(P_rc/P_r)
        nr = P_rc/P_r
        if (rounded > nr):
            P_n = P_c -1/total_count
        else:
            P_n = P_c
        mutinf = math.log10(P_n/(P_r*P_c))
        #print([P_rc, P_c, P_r, P_n])
    else:
        mutinf = math.log10(P_rc/(P_r*P_c))
    return ([target_word1, target_word2, mutinf])

def count(ci, word):
   
    offsets = ci.offsets(word)

    return len(offsets)

def pmi(text):
    '''
    iterates through and finds shit
    '''
    
    clean_doc = clean_text(text.split())
    total_count = len(clean_doc)
    
    test_ci = make_ci(clean_doc)
    
    pmi_list = []
    ordered_set_hold = []
    ordered_set = [i for i in clean_doc if i not in ordered_set_hold and len(i) > 0]

    
    index = 0
    #realistically only words that at some point occur in a 3 word window are really worth looking at esp with these short texts
    for i in enumerate(ordered_set):
        if i[0] < (len((ordered_set))-1):
            item = mutual_informativity(test_ci, i[1], ordered_set[i[0] + 1], total_count)
            if(item not in pmi_list) & (item[0] != item[1]):
                pmi_list.append(item)
        if i[0] < (len(ordered_set)-2):
            item = mutual_informativity(test_ci, i[1], ordered_set[i[0] + 2], total_count)
            if(item not in pmi_list) & (item[0] != item[1]):
                pmi_list.append(item)        
        if i[0] < (len(ordered_set)-3):
            item = mutual_informativity(test_ci, i[1], ordered_set[i[0] + 3], total_count)
            if(item not in pmi_list) & (item[0] != item[1]):
                pmi_list.append(item)
        else:
            return(pmi_list)
        
#just to allow for sorting by actual pmi index
def takeSecond(elem):
    #print(elem[2])
    return elem[2]

def pmi_high(pmi_output, n):
    cat = pmi_output
    cat.sort(key = takeSecond, reverse = True)
    return cat[:n]

def pmi_low(pmi_output, n):
    cat = pmi_output
    cat.sort(key = takeSecond, reverse = False)
    return cat[:n]

def generate_lexicals(input_value, n = 2000):
    pmi_text = pmi(input_value)
    word_pairs = []
    item_dis = []
    latent_dis = []
    x = 0
    for i in pmi_high(pmi_text, n):
        if i != None:
            j, k , l = latent_meaning_spacy([i[0], i[1]])
            word_pairs.append(j)
            item_dis.append(k)
            latent_dis.append(l)
            #bar update
            
    latent_meanings = pd.DataFrame({
    
    'word_pairs': word_pairs,
    'item_dis': item_dis,
    'latent_dis': latent_dis
    
    })
    latent_meanings['difference']= latent_meanings['item_dis'] - latent_meanings['latent_dis']
    lexicals = latent_meanings[(latent_meanings['difference'] > .05) & (latent_meanings['item_dis'] > 0.05)].sort_values(by = 'difference', ascending = False)
    return( ''.join(list(lexicals['word_pairs'])))


navbar = dbc.Navbar(
    [
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(dbc.NavbarBrand("Madeline Kehl", className="ml-2")),
                ],
                align="center",
                no_gutters=True,
            ),
            href="https://www.github.com/madkehl/Tessa",
        ),
        dbc.NavbarToggler(id="navbar-toggler"),
    ],)

def description_card():
    """
    OUTPUT: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.H5(""),
            html.Br(),
            html.H3("Recommendations based on collaborative filtering techniques, Starbucks Data"),
            html.Br(),
            html.A('This web app is designed to be a clean and simple way of visualizing the results of a rather complex analysis. The goal of this was to understand what offers would be most impactful for which demographics.  Information such as age, income, gender and years member was available for analysis as well as offers and their various characteristics.  More behind-the-scenes data cleaning and analysis, as well as other intermediary discussion is available in the Github for the project and the Jupyter notebook rendering.'),
            html.A("If you love this, please visit the full project either by clicking my name above (my Github) or viewing the full notebook here.", href = 'https://nbviewer.jupyter.org/github/madkehl/Starbucks/blob/main/web_app/models/Starbucks_full_documentation.ipynb'),
            html.Div(
                id="intro",
                children=[
                    html.Br(),
                    html.A("This interactive graph allows you to examine recommendations by demographic.  Double click labels in the legend to view demographics of interest, and use the offer selection bar to filter by offer types."),
                    html.Br(),
                    html.A('If no offers exist that meet all the criteria selected, then you will simply see all results.'),
                ],
            ),
        ],
    )


starter = generate_lexicals(test_doc)
#app structure

app.layout = html.Div(
    id="app-container",
    children=[
        # Left column
        html.Div(id="left-column", className="four columns", children=[description_card()]),
        dcc.Input(id='text-div', value= test_doc, type='text'),
        
        html.Button(id='submit-button', type='submit', children='Submit'),
        html.Div(id='output_div', children = [starter]),
        html.Br(),
        ],
    style={'marginBottom': 50, 'marginLeft': 25,'marginRight':25, 'marginTop': 25},
)

#this connects user interactions with backend code
@app.callback(
     Output("output_div", "children"),
    [Input('text-div', 'value'),
    Input('submit-button', 'n_clicks')],
)

def update_output(clicks, input_value):
    
    if clicks is not None:
        return(generate_lexicals(input_value))



def main():
    '''
    as the main function this runs whenever the file is called
    
    it sets the port and then runs the app through the desired port
    '''
    
    if len(sys.argv) == 2: 
        from waitress import serve
        serve(server, host="0.0.0.0", port=8080)
    else:
        port = int(os.environ.get('PORT', 5000))
        app.run_server(host='0.0.0.0', port=port)
  



# Run the server
if __name__ == "__main__":
    main()
  

    
