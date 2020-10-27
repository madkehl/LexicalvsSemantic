##borrowed from https://github.com/plotly/dash-sample-apps/tree/master/apps/dash-clinical-analytics
#key imports
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, ClientsideFunction, State
import dash_bootstrap_components as dbc
import sys
import numpy as np
import math

import spacy
nlp = spacy.load("en_core_web_md")
n_vectors = 5000  # number of vectors to keep
removed_words = nlp.vocab.prune_vectors(n_vectors)

vocab = nlp.vocab.strings


from string import punctuation
from re import sub
punctuation = punctuation +'”'+'“'+'’' + '—' + '’' + '‘' +'0123456789'

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
'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'neither', 'upon', 'first', 'second', 'third']

test_doc = 'Your entry here'

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
    ms = list(zip(ms[0][0], ms[1][0], ms[2][0]))
  #  print(ms)
    ms = [i for i in ms if str(nlp.vocab.strings[i[0]]).lower() != word]
    ms = [i for i in ms if str(nlp.vocab.strings[i[0]]).lower() + 's' != word]
    ms = [i for i in ms if str(nlp.vocab.strings[i[0]]).lower() + 'es' != word]
    words = [nlp.vocab.strings[w[0]] for w in ms]
    distances = [i[2] for i in ms]
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
        
#just to allow for sorting by actual pmi index
def pair_words(text):
    '''
    iterates through and finds shit
    '''
    
    clean_doc = clean_text(text.split())
    
    words_pairs = []

    index = 0
    #realistically only words that at some point occur in a 3 word window are really worth looking at esp with these short texts
    for i in enumerate(clean_doc):
        if i[0] < (len((clean_doc))-1):
            item = (i[1], clean_doc[i[0] + 1])
            if(item not in words_pairs) & ((clean_doc[i[0] + 1], i[1]) not in words_pairs) & (item[0] != item[1]):
                words_pairs.append(item)
        if i[0] < (len(clean_doc)-2):
            item =  (i[1], clean_doc[i[0] + 2])
            if(item not in words_pairs) & ((clean_doc[i[0] + 2], i[1]) not in words_pairs) & (item[0] != item[1]):
                words_pairs.append(item)        
        if i[0] < (len(clean_doc)-3):
            item =  (i[1], clean_doc[i[0] + 3])
            if(item not in words_pairs) & ((clean_doc[i[0] + 3], i[1]) not in words_pairs) & (item[0] != item[1]):
                words_pairs.append(item)
        else:
            return(words_pairs)

def generate_lexicals(input_value):
    if isinstance(input_value, str):
        pmi_text = pair_words(input_value)
    else:
        return('error')

    word_pairs = []
    item_dis = []
    latent_dis = []

    if pmi_text != None:
        n = len(pmi_text)
        for i in (pmi_text):
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
        lexicals = latent_meanings[(latent_meanings['difference'] > 0) & (latent_meanings['item_dis'] > 0)].sort_values(by = 'difference', ascending = False)
        lexicals_temp = [str(i) for i in lexicals['word_pairs']]
    else:
        return('no text entered')
    if len(lexicals_temp) > 0:
        fig = go.Figure(data=[go.Table(
                header=dict(values=list(lexicals.columns),
                    fill_color='paleturquoise',
                    align='left'),
                cells=dict(values=[lexicals.word_pairs, lexicals.item_dis, lexicals.latent_dis, lexicals.difference],
                   fill_color='lavender',
                   align='left'))
            ])



        
        return(dcc.Graph(figure=fig))
    else:
        return('no relevant pairs in this selection')


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
            html.H3("Lexically related words"),
            html.Br(),
            html.A('The code driving this web app was originally developed to assist in the development of experimental items in studies related to aphasia.  The goal is to determine a subset of words that are likely to be lexically related, instead of just semantically.  The way our algorithm does this is by following the infographic below.  Please note that this process is not the most efficient and texts of more than a few hundred words, will take an increasingly long time to run.'),
            html.A("If you love this, please visit the full project either by clicking my name above (my Github) or viewing the full notebook here.", href = 'https://nbviewer.jupyter.org/github/madkehl/Starbucks/blob/main/web_app/models/Starbucks_full_documentation.ipynb'),
            html.Div(
                id="intro",
                children=[
                    html.Br(),
                    html.A("Please enter text of interest here.  While the code is running, a loading graph icon will appear.  Don't worry if it is taking a long time, as long as the graph is moving, it's working."),
                    html.Br(),
                    html.A('It will then display the words from your selection that are most likely to be a lexical pairing, rather than a semantic one. If none exist, it will print a corresponding statement.  To use again, please just refresh the page.'),
                    html.Br(),
                ],
            ),
        ],
    )



starter = generate_lexicals(test_doc)
#app structure

app.layout = html.Div(
    id="app-container",
    children=[
        html.Div(id="nav", children = [navbar]),
        html.Div(id="left-column", className="four columns", children=[description_card()]),
        dcc.Input(id='text-div', value= test_doc, type='text'),
        html.Br(),
        dcc.Loading(id = "loading-icon", 
                children=[html.Div(id='output_div', children = [starter])], type="graph"),
        html.Br(),
        html.Button(id='submit-button', type='submit', children='Submit'),
        html.Br()
      
        ],
    style={'marginBottom': 50, 'marginLeft': 25,'marginRight':25, 'marginTop': 25},
)

#this connects user interactions with backend code
@app.callback(
      Output("output_div", "children"),
    [ Input('submit-button', 'n_clicks'),
      Input('text-div', 'value'),],
)

def update_output(clicks, input_value):

    ctx = dash.callback_context
    
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "submit-button":
    #if clicks is not None:
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
  

    
