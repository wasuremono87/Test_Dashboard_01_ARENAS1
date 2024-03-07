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
        html.H1("Keywords & n-Grams", style={'textAlign': 'center'}),
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

    html.Div([
        # Left panel for settings - ngrams
        html.Div([
            html.H4("Settings - n-Grams"),
            dcc.Dropdown(id='ngram-type-selector', options=[
                {'label': 'Bigramme', 'value': 'Texts_Lemmatized_Bigrams'},
                {'label': 'Trigramme', 'value': 'Texts_Lemmatized_Trigrams'},
                {'label': 'Bigramme gefiltert', 'value': 'Texts_Lemmatized_Bigrams_Filtered'},
                {'label': 'Trigramme gefiltert', 'value': 'Texts_Lemmatized_Trigrams_Filtered'}
            ], multi=True, placeholder="n-Gramm Typ/en"),
            html.Label('Anzahl der anzuzeigenden n-Gramme:'),
            dcc.Slider(id='num-ngrams-slider', min=1, max=50, value=10, marks={i: str(i) for i in range(1, 51, 5)}, step=1)
        ], style={'width': '20%', 'display': 'inline-block', 'verticalAlign': 'top'}),

        # Right panel for Worthäufigkeiten visualizations
        html.Div([
            html.H4("Visualizations - n-Grams"),
            dcc.Graph(id='ngram-freq-graph'),
            html.H4("n-Gram Network"),
            dcc.Graph(id='ngram-network-graph')
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
            html.H4("Visualizations - Word Embeddings"),
            html.Div(id='search-results-container') 
        ], style={'width': '80%', 'display': 'inline-block'}),
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

@app.callback(
    Output('ngram-freq-graph', 'figure'),
    [Input('text-selection-mode', 'value'),
     Input('global-text-selector', 'value'),
     Input('ngram-type-selector', 'value'),
     Input('num-ngrams-slider', 'value')]  # Füge den Slider als Input hinzu
)

def update_ngram_graph(text_selection_mode, selected_texts, selected_ngram_types, num_ngrams):
    # Wenn "Alle Texte" ausgewählt ist, überschreibe selected_texts mit allen verfügbaren Texten
    if text_selection_mode == 'ALL_TEXTS':
        selected_texts = list(data['Texts_Lemmatized_ALL'].keys())
    
    if not selected_texts or not selected_ngram_types:
        return px.bar()
    
    all_ngrams = []
    # Sammle alle n-Gramme basierend auf den ausgewählten Texten
    for ngram_type in selected_ngram_types:
        for text in selected_texts:
            ngrams = data.get(ngram_type, {}).get(text, [])
            ngrams_final = []
            for ngram in ngrams:
                words = ngram.split(" ")
                if ngram_type.startswith("Texts_Lemmatized_Bi"):
                    if words[0] not in set(stopwords) and words[1] not in set(stopwords):
                        ngrams_final.append(ngram)
                else:
                    if words[0] not in set(stopwords) and words[1] not in set(stopwords) and words[2] not in set(stopwords):
                        ngrams_final.append(ngram)
            all_ngrams.extend(ngrams_final)
                    
    
    df = pd.DataFrame(all_ngrams, columns=['n-Gramm'])
    freq = df['n-Gramm'].value_counts().reset_index()
    freq.columns = ['n-Gramm', 'Häufigkeit']
    
    # Beschränke die Anzeige auf die Top-N n-Gramme, basierend auf dem Slider-Wert
    freq_top_n = freq.head(num_ngrams)
    
    # Erstelle das Balkendiagramm, diesmal horizontal für bessere Lesbarkeit
    fig = px.bar(freq_top_n, x='Häufigkeit', y='n-Gramm', orientation='h', title='Häufigkeit von n-Grammen')
    fig.update_layout(yaxis={'categoryorder':'total ascending'})  # Häufigstes n-Gramm oben
    
    return fig

########### NEU NETZWERKDARSTELLUNG ##############
# Funktion zur Erstellung des Netzwerkdiagramms
def create_network_graph(selected_ngram_types, selected_texts, data, num_ngrams):
    G = nx.Graph()
    
    all_ngrams = []
    # Sammle alle n-Gramme basierend auf den ausgewählten Texten
    for ngram_type in selected_ngram_types:
        for text in selected_texts:
            ngrams = data.get(ngram_type, {}).get(text, [])
            ngrams_final = []
            for ngram in ngrams:
                words = ngram.split(" ")
                if ngram_type.startswith("Texts_Lemmatized_Bi"):
                    if words[0] not in set(stopwords) and words[1] not in set(stopwords):
                        ngrams_final.append(ngram)
                else:
                    if words[0] not in set(stopwords) and words[1] not in set(stopwords) and words[2] not in set(stopwords):
                        ngrams_final.append(ngram)
            all_ngrams.extend(ngrams_final)
    
    df = pd.DataFrame(all_ngrams, columns=['n-Gramm'])
    freq = df['n-Gramm'].value_counts().reset_index()
    freq.columns = ['n-Gramm', 'Häufigkeit']
    
    # Beschränke die Anzeige auf die Top-N n-Gramme, basierend auf dem Slider-Wert
    freq_top_n = freq.head(num_ngrams)
    
    # Ein Dictionary zur Speicherung der Häufigkeit jeder Kante (später benötigt zur berechnung der kantendicke)
    edge_counts = defaultdict(int)
    
    # Hinzufügen der Kanten für jedes n-Gramm
    for ngram in freq_top_n["n-Gramm"].tolist():
        words = ngram.split(" ")  # Splitte jedes n-Gramm in einzelne Wörter
        # Für jedes n-Gramm, generiere alle möglichen Bigramme (in diesem Fall Verbindungen zwischen aufeinanderfolgenden Wörtern) und füge sie als Kanten hinzu
        for i in range(len(words)-1):
            G.add_edge(words[i], words[i+1])
            # ... und dictionary der edge_counts befüllen
            edge = (words[i], words[i + 1])
            edge_counts[edge] += 1
            
    # Positionen für die Knoten
    pos = nx.spring_layout(G)

    # Erstelle die Kanten- und Knoten-Traces
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    node_x, node_y = [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

    ######## Aussehen der Kanten und Knoten
    ########## KANTEN

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), mode='lines')

    ########## KNOTEN
    # Zählen, wie oft jedes Wort vorkommt + Erstellen einer Liste mit der Größe für jeden Knoten basierend auf der Wort-Häufigkeit
    word_counts = Counter([word for ngram in all_ngrams for word in ngram.split(" ")])
    node_sizes = [word_counts[node]*10 for node in G.nodes()]  # Multipliziert mit 10 für die Sichtbarkeit
    node_labels = [f"{node}\n(n={word_counts[node]})" for node in G.nodes()]
    node_frequencies = [word_counts[node] for node in G.nodes()]

    # Erstellen einer Farbskala von Hellgrün zu Dunkelgrün basierend auf der Häufigkeit
    # Normalisieren der Frequenzen für die Farbskala
    normalized_frequencies = np.array(node_frequencies) / max(node_frequencies)

    # Funktion zur linearen Interpolation zwischen Grau und Grün
    def interpolate_color(freq):
        # Grau (136, 136, 136) zu Grün (0, 255, 0)
        start_color = np.array([136, 136, 136])
        end_color = np.array([0, 255, 0])
        interpolated_color = start_color + (end_color - start_color) * freq
        return f"rgb({int(interpolated_color[0])}, {int(interpolated_color[1])}, {int(interpolated_color[2])})"

    colors = [interpolate_color(freq) for freq in normalized_frequencies]
    
    node_trace = go.Scatter(
        x=node_x, 
        y=node_y, 
        mode='markers+text',  # Modus auf 'markers+text' setzen, um Text direkt anzuzeigen
        #marker=dict(size=node_sizes, color=colors, showscale=True), 
        text=node_labels,  # Knotennamen als Text
        textposition="top center",  # Position des Textes relativ zu den Markern
        hoverinfo='text'
    )
    
    return go.Figure(data=[edge_trace, node_trace])

# Callback zur Aktualisierung des Netzwerkdiagramms
@app.callback(
    Output('ngram-network-graph', 'figure'),
    [Input('text-selection-mode', 'value'),
     Input('global-text-selector', 'value'),
     Input('ngram-type-selector', 'value'),
     Input('num-ngrams-slider', 'value')]
)

def update_network_graph(text_selection_mode, selected_texts, selected_ngram_types, num_ngrams):
    if text_selection_mode == 'ALL_TEXTS':
        selected_texts = list(data['Texts_Lemmatized_ALL'].keys())
        
    if not selected_texts or not selected_ngram_types:
        return go.Figure()
    
    fig = create_network_graph(selected_ngram_types, selected_texts, data, num_ngrams)
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
    
    results_html = []
    for left_context, keyword, right_context in zip(left_contexts_f, keywords_f, right_contexts_f):
        # Erstelle ein Flex-Container-Div für jedes Ergebnis
        result_html = html.Div([
            html.Div(left_context, style={'flex': 1, 'text-align': 'right', 'margin-right': '10px', 'color': 'grey'}),  # Rechtsbündiger linker Kontext
            html.Div(keyword, style={'flex': 0, 'font-weight': 'bold'}),  # Zentralisiertes, fettgedrucktes Keyword
            html.Div(right_context, style={'flex': 1, 'text-align': 'left', 'margin-left': '10px', 'color': 'grey'}),  # Linksbündiger rechter Kontext
        ], style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center'})
        
        results_html.append(result_html)
    
    return html.Div(results_html, style={'margin': '20px'})  # Container für alle Ergebnisse


####


if __name__ == '__main__':
    app.run_server(debug=True)
