import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from collections import Counter
import numpy as np
from collections import defaultdict
from ast import literal_eval

import plotly.express as px
import pandas as pd
import plotly.graph_objs as go
from matplotlib import pyplot as plt
import networkx as nx
import os, sys
module_path = os.path.abspath(os.path.join('../src'))
if module_path not in sys.path:
    sys.path.insert(0, module_path)
from we_model import We_model

##############################################################

# Daten einlesen und aufbereiten als Dictionary
data_filepath = '../data/potsdamer_treffen/potsdamer_text_data.csv'

df = pd.read_csv(data_filepath, sep="\t", dtype={'datum': str})

df['Texts_Tokenized_ALL'] = df['Texts_Tokenized_ALL'].apply(lambda x: literal_eval(x)).tolist()
df['Texts_Lemmatized_ALL'] = df['Texts_Lemmatized_ALL'].apply(lambda x: literal_eval(x)).tolist()
df['Texts_Lemmatized_NOUN'] = df['Texts_Lemmatized_NOUN'].apply(lambda x: literal_eval(x)).tolist()
df['Texts_Lemmatized_VERB'] = df['Texts_Lemmatized_VERB'].apply(lambda x: literal_eval(x)).tolist()
df['Texts_Lemmatized_ADJ'] = df['Texts_Lemmatized_ADJ'].apply(lambda x: literal_eval(x)).tolist()
df['Texts_Lemmatized_XY'] = df['Texts_Lemmatized_XY'].apply(lambda x: literal_eval(x)).tolist()
df['Texts_Lemmatized_Bigrams'] = df['Texts_Lemmatized_Bigrams'].apply(lambda x: literal_eval(x)).tolist()
df['Texts_Lemmatized_Trigrams'] = df['Texts_Lemmatized_Trigrams'].apply(lambda x: literal_eval(x)).tolist()
df['Texts_Lemmatized_Bigrams_Filtered'] = df['Texts_Lemmatized_Bigrams_Filtered'].apply(lambda x: literal_eval(x)).tolist()
df['Texts_Lemmatized_Trigrams_Filtered'] = df['Texts_Lemmatized_Trigrams_Filtered'].apply(lambda x: literal_eval(x)).tolist()
df['Texts_POS'] = df['Texts_POS'].apply(lambda x: literal_eval(x)).tolist()

texts = df["text"].unique()
texts = [text.replace(".txt", "") for text in texts]
modelpaths = {text: "../data/word_embedding/" + text + "we_model.word_vectors" for text in texts}
modelpaths["all_texts"] = "../data/word_embedding/all_texts_we_model.word_vectors"

dict_all_tok = {}
dict_all = {}
dict_noun = {}
dict_verb = {}
dict_adj = {}
dict_xy = {}
dict_bi = {}
dict_tri = {}
dict_bi_filt = {}
dict_tri_filt = {}
dict_pos_tags = {}

for index, row in df.iterrows():
    text = row["text"]
    text = text.replace(".txt", "") 
    
    tok_all = row['Texts_Tokenized_ALL']
    lem_all = row['Texts_Lemmatized_ALL']
    lem_noun = row['Texts_Lemmatized_NOUN']
    lem_verb = row['Texts_Lemmatized_VERB']
    lem_adj = row['Texts_Lemmatized_ADJ']
    lem_xy = row['Texts_Lemmatized_XY']
    lem_bi = row['Texts_Lemmatized_Bigrams']
    lem_tri = row['Texts_Lemmatized_Trigrams']
    lem_bi_filt = row['Texts_Lemmatized_Bigrams_Filtered']
    lem_tri_filt = row['Texts_Lemmatized_Trigrams_Filtered']
    pos_tag = row["Texts_POS"]

    dict_all_tok[text] = tok_all
    dict_all[text] = lem_all
    dict_noun[text] = lem_noun
    dict_verb[text] = lem_verb
    dict_adj[text] = lem_adj
    dict_xy[text] = lem_xy
    dict_bi[text] = lem_bi
    dict_tri[text] = lem_tri
    dict_bi_filt[text] = lem_bi_filt
    dict_tri_filt[text] = lem_tri_filt
    dict_pos_tags[text] = pos_tag

#dict_all_tok["all_texts"] = [value for key, value in dict_all_tok.items()]
dict_all_tok["all_texts"] = [token for value in dict_all_tok.values() for token in value]


data = {
    "Texts_Tokenized_ALL":dict_all_tok,
    "Texts_Lemmatized_ALL": dict_all,
    "Texts_Lemmatized_NOUN": dict_noun,
    "Texts_Lemmatized_VERB": dict_verb,
    "Texts_Lemmatized_ADJ": dict_adj,
    "Texts_Lemmatized_XY": dict_xy,
    "Texts_Lemmatized_Bigrams": dict_bi,
    "Texts_Lemmatized_Trigrams": dict_tri,
    "Texts_Lemmatized_Bigrams_Filtered": dict_bi_filt,
    "Texts_Lemmatized_Trigrams_Filtered": dict_tri_filt,
    "Texts_POS": dict_pos_tags,
    "Models": modelpaths,
}

# Initialisiere das WE Modell mit dem neuen Pfad
we_model = We_model(data_Texts_Lemmatized_ALL=data["Texts_Lemmatized_ALL"], modelpath=data["Models"]["all_texts"])

# Einlesen der Stopwortliste
with open("../data/stopwords-de.txt", 'r', encoding='utf-8') as file:
    stopwords = [line.strip() for line in file.readlines()]
##############################################################


app = dash.Dash(__name__)
server = app.server


app.layout = html.Div([
    # Container for the entire layout
    html.Div([
        html.H1("Keywords", style={'textAlign': 'center'}),
        # Left panel for settings - Worthäufigkeiten
        html.Div([
            html.H4("Settings - Keywords"),
            dcc.RadioItems(id='text-selection-mode', options=[
                {'label': 'Alle Texte', 'value': 'ALL_TEXTS'},
                {'label': 'Texte auswählen', 'value': 'SELECT_TEXTS'}
            ], value='ALL_TEXTS'),
            
            html.Hr(),
            dcc.Dropdown(id='global-text-selector', options=[{'label': key, 'value': key} for key in set(data['Texts_Lemmatized_ALL'].keys())],
                         multi=True, placeholder="Textauswahl", style={'display': 'none'}),
            html.Hr(),
            html.Label('Anzahl der anzuzeigenden Wörter:'),
            dcc.Slider(id='num-words-slider', min=1, max=100, value=20, marks={i: str(i) for i in range(1, 101, 10)}, step=1),
            
            html.Hr(),
            html.H4("Keywords aus dem Text filtern:"),
            dcc.Dropdown(id='pos-selector', options=[{'label': tag, 'value': tag} for tag in ["ALL", "NOUN", "VERB", "ADJ"]], value=["ALL"], multi=True, placeholder="Filter"),
        ], style={'width': '20%', 'display': 'inline-block', 'verticalAlign': 'top'}),

        # Right panel for Worthäufigkeiten visualizations
        html.Div([
            html.H4("Visualizations - Keywords"),
            dcc.Graph(id='word-freq-graph'),
        ], style={'width': '80%', 'display': 'inline-block'}),
    ]),
    
    # Divider line
    html.Hr(),
    
    html.H1("Word Embeddings", style={'textAlign': 'center'}),
    # Container for the Word Embeddings section
    html.Div([
        # Left panel for settings - Word Embeddings
        html.Div([
            html.H4("Settings - Word Embeddings"),
            dcc.RadioItems(id='model-selection-mode', options=[
                {'label': 'Alle Texte', 'value': 'ALL_TEXTS'},
                {'label': 'Texte auswählen', 'value': 'SELECT_TEXTS'}
            ], value='ALL_TEXTS'),
            
            html.Hr(),
            dcc.Dropdown(id='model-selector', options=[{'label': model_name, 'value': model_name} for model_name in modelpaths.keys()], value='all_texts', multi=False, placeholder="Wähle ein WE-Model"),

            html.Hr(),
            dcc.Dropdown(id='word-dropdown', options=[{'label': word, 'value': word} for word in we_model.get_model_vocabulary().keys() if word not in stopwords], value=[], multi=True, placeholder="Wähle Keywords"),
            
            html.Hr(),
            dcc.RadioItems(id='dimension-selector', options=[
                {'label': '2D', 'value': '2d'},
                {'label': '3D', 'value': '3d'}
            ], value='2d'),
            
            html.Hr(),
            html.Label('Anzahl der nächsten Nachbarn:'),
            dcc.Slider(id='topn-slider', min=1, max=20, value=10, marks={i: str(i) for i in range(1, 21)}, step=1)
        ], style={'width': '20%', 'display': 'inline-block', 'verticalAlign': 'top'}),

        # Right panel for Word Embeddings visualizations
        html.Div([
            html.H4("Visualizations - Word Embeddings"),
            dcc.Graph(id='word-graph'),
             # For displaying search results
        ], style={'width': '80%', 'display': 'inline-block'}),
    ]),

    html.Div([
        # Left panel for settings - Word Embeddings
        html.Div([
            html.H4("Keyword in Context"),
            dcc.Input(id='keyword-input', type='text', placeholder='Gib ein Keyword ein'),
            html.Button('Suche', id='search-button')
        ], style={'width': '20%', 'display': 'inline-block', 'verticalAlign': 'top'}),
        
        # Right panel for Word Embeddings visualizations
        html.Div([
        ], style={'width': '80%', 'display': 'inline-block', 'textAlign': 'center'}),
    ]),
    
    html.Div([
        html.H4("Results", style={'textAlign': 'center'}),
        html.Div(id='search-results-container') 
    ]),
    
], style={'fontFamily': 'sans-serif'})

@app.callback(
    Output('global-text-selector', 'style'),
    [Input('text-selection-mode', 'value')]
)

def toggle_text_selector(selection_mode):
    if selection_mode == 'SELECT_TEXTS':
        return {}  # Zeige das Dropdown-Menü
    else:
        return {'display': 'none'}  # Verstecke das Dropdown-Menü

@app.callback(
    Output('word-freq-graph', 'figure'),
    [Input('text-selection-mode', 'value'),
     Input('global-text-selector', 'value'),
     Input('pos-selector', 'value'),
     Input('num-words-slider', 'value')]  # Füge den Slider als Input hinzu
)

def update_word_graph(selection_mode, selected_texts, selected_pos_tags, num_words):
    if selection_mode == 'ALL_TEXTS':
        # Verwende alle Texte, wenn "Alle Texte" ausgewählt ist
        texts_to_use = list(data['Texts_Lemmatized_ALL'].keys())
    else:
        # Verwende ausgewählte Texte, wenn "Texte auswählen" aktiv ist
        texts_to_use = selected_texts if selected_texts else []
    
    # Führe die Filterung basierend auf den ausgewählten POS-Tags durch
    filtered_words = []
    for pos_tag in selected_pos_tags:
        key = f"Texts_Lemmatized_{pos_tag}"
        for text in texts_to_use:
            tokens = data.get(key, {}).get(text, [])
            filtered_words.extend(tokens)

    # Entferne Stopwords
    filtered_words = [word for word in filtered_words if word not in set(stopwords)]
    
    df = pd.DataFrame(filtered_words, columns=['Wort'])
    freq = df['Wort'].value_counts().reset_index()
    freq.columns = ['Wort', 'Häufigkeit']

     # Sortiere die Frequenztabelle und beschränke sie auf die Top N Wörter
    top_words = freq.sort_values(by='Häufigkeit', ascending=False).head(num_words)

    # Erstelle ein horizontal ausgerichtetes Balkendiagramm
    fig = px.bar(top_words, x='Häufigkeit', y='Wort', orientation='h')
    fig.update_layout(yaxis={'categoryorder':'total ascending'})  # Häufigstes Wort oben

    return fig

#####################################
#### Hinzufügen von EMBEDDINGS ######
#####################################

@app.callback(
    Output('word-graph', 'figure'),
    [Input('model-selector', 'value'), 
     Input('word-dropdown', 'value'),
     Input('dimension-selector', 'value'),
     Input('topn-slider', 'value')] 
)

def update_word_embedding_graph(selected_model, selected_words, selected_dimension, topn):
    # Wenn "ALL_TEXTS" ausgewählt ist, lade das allgemeine Modell
    if selected_model == "ALL_TEXTS":
        modelpath = modelpaths["all_texts"]
    else:
        # Ansonsten lade das spezifische Modell basierend auf der Auswahl
        modelpath = modelpaths.get(selected_model, modelpaths["all_texts"])
    
    we_model = We_model(data_Texts_Lemmatized_ALL=data["Texts_Lemmatized_ALL"], modelpath=modelpath)

    # Verwende deine Methode zur Visualisierung der nächsten Nachbarn
    fig = we_model.visualize_nearest_neighbours(dimensions=selected_dimension, keyword_list=selected_words, topn=topn, stopwords=stopwords)
    return fig

@app.callback(
    [Output('model-selector', 'style'),
     Output('model-selector', 'value')],
    [Input('model-selection-mode', 'value')]
)

def toggle_model_selector(selection_mode):
    if selection_mode == 'SELECT_TEXTS':
        # Zeige das Dropdown-Menü und setze den Standardwert zurück
        return {}, "all_texts"  # Ersetze "all_texts" mit dem Standardwert deiner Wahl
    else:
        # Verstecke das Dropdown-Menü und wähle "ALL_TEXTS" als Modell
        return {'display': 'none'}, "ALL_TEXTS"

@app.callback(
    Output('search-results-container', 'children'),
    [Input('search-button', 'n_clicks')],
    [State('keyword-input', 'value'),
     State('model-selection-mode', 'value'),
     State('model-selector', 'value')]
)
def update_search_results(n_clicks, keyword, selection_mode, selected_model):
    if not n_clicks or not keyword:
        return 'Bitte gib ein Keyword ein und starte die Suche.'

    # Entscheide, welches Textset basierend auf der Auswahl benutzt wird
    if selection_mode == 'ALL_TEXTS':
        texts_to_search = data["Texts_Tokenized_ALL"]["all_texts"]
    else:
        #texts_to_search = data["Texts_Tokenized_ALL"].get(selected_model, [])
        texts_to_search = data["Texts_Tokenized_ALL"][selected_model]
        
    # Suche nach dem Keyword und extrahiere den Kontext
    results = []
    for i, token in enumerate(texts_to_search):
        if token == keyword:
            left_context = texts_to_search[max(i-20, 0):i]  # Angenommener Kontext von 5 Token
            right_context = texts_to_search[i+1:i+21]
            results.append(' '.join(left_context + [token] + right_context))
    
    if not results:
        return 'Keine Ergebnisse gefunden.'
        

    left_contexts = []
    keywords = []
    right_contexts = []
    
    for result in results:
        left_context, keyword, right_context = result.partition(keyword)  # Teile den String in Kontext und Keyword
        left_contexts.append(left_context)
        keywords.append(keyword)
        right_contexts.append(right_context)

    zu_entfernen = set()
    for i, kontext_i in enumerate(left_contexts):
        for j, kontext_j in enumerate(left_contexts):
            if i != j and kontext_i.endswith(kontext_j):
                zu_entfernen.add(j)
            elif i != j and kontext_j.endswith(kontext_i):
                zu_entfernen.add(i)
    
    left_contexts_f = []
    keywords_f = []
    right_contexts_f = []
    
    for i, kontext in enumerate(left_contexts):
        if i not in zu_entfernen:
            left_contexts_f.append(left_contexts[i])
            keywords_f.append(keywords[i])
            right_contexts_f.append(right_contexts[i])
    
    '''
    results_html = []
    for left_context, keyword, right_context in zip(left_contexts_f, keywords_f, right_contexts_f):
        # Erstelle ein Flex-Container-Div für jedes Ergebnis
        result_html = html.Div([
            html.Div(left_context, style={'flex': 1, 'text-align': 'right', 'margin-right': '10px', 'color': 'grey'}),  # Rechtsbündiger linker Kontext
            html.Div(keyword, style={'flex': 0, 'font-weight': 'bold', 'color': 'red'}),  # Zentralisiertes, fettgedrucktes Keyword
            html.Div(right_context, style={'flex': 1, 'text-align': 'left', 'margin-left': '10px', 'color': 'grey'}),  # Linksbündiger rechter Kontext
        ], style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center'})
        
        results_html.append(result_html)
    
    return html.Div(results_html, style={'margin': '20px'})  # Container für alle Ergebnisse
    '''

    results_html = [html.Tr([
        html.Td(left_context, style={'text-align': 'right', 'color': 'grey'}), 
        html.Td(keyword, style={'font-weight': 'bold', 'color': 'red'}),
        html.Td(right_context, style={'text-align': 'left', 'color': 'grey'})
    ]) for left_context, keyword, right_context in zip(left_contexts_f, keywords_f, right_contexts_f)]
    
    # Wrap the rows in a table structure
    results_table = html.Table(
        # Table body
        [html.Tbody(results_html)],
        # Table styling
        style={'width': '100%', 'margin': '20px'}
    )
    
    return results_table


####


if __name__ == '__main__':
    app.run_server(debug=True)
