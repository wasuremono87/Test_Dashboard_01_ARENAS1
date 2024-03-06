from gensim.models import Word2Vec, KeyedVectors
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objs as go
import csv
import os, sys

class We_model():
    '''
    # Erstellt eine Word-Embedding-Modell auf der Grundlage einer Liste von lemmatata eines Textes
    # data["Texts_Lemmatized"] = [[Text_1_Lemma_1, Text1_Lemma_2, ...], [Text_2_Lemma_1, Text_2_Lemma2, ...], ....]
    def __init__(self, data, modelpath):

        if os.path.exists(modelpath):
            model = KeyedVectors.load_word2vec_format(modelpath, binary=False)
        else:
            model = Word2Vec(sentences=data["Tokens_Lemmatized_ALL"], vector_size=300, window=5, min_count=10, workers=4)
            word_vectors = model.wv
            word_vectors.save_word2vec_format(modelpath, binary=False)
            model = KeyedVectors.load_word2vec_format(modelpath, binary=False)
        
        self.model = model
        self.modelpath = modelpath
    '''
    
    def __init__(self, data_Texts_Lemmatized_ALL, modelpath): # data_Texts_Lemmatized_ALL = Liste mit Listen von Strings (Texte)
        if os.path.exists(modelpath):
            print("Lade existierendes Modell...")
            model = KeyedVectors.load_word2vec_format(modelpath, binary=False)
        else:
            print("Trainiere neues Modell...")
            try:
                model = Word2Vec(sentences=data_Texts_Lemmatized_ALL, vector_size=300, window=5, min_count=2, workers=4)
                model.wv.save_word2vec_format(modelpath, binary=False)
                model = KeyedVectors.load_word2vec_format(modelpath, binary=False)
                print("Modell erfolgreich trainiert und gespeichert.")
            except Exception as e:
                print(f"Fehler beim Trainieren des Modells: {e}")
                model = None
                
        self.model = model
        self.modelpath = modelpath

    
    def get_model_vocabulary(self):
        vocab = self.model.key_to_index
        return vocab
        
    def get_nearest_neigbours(self, keyword, topn):
    
        # Ausgabe von topn (z.B. 10) ähnlicher Vektoren vom keyword-Vektor
        sims = self.model.most_similar(keyword, topn=topn)
        
        return sims

    def compute_vectors(self, keyword_list_1, keyword_list_2):
        # In diesem Block kann man mit den Word-Embeddings Vektorberechnungen durchführen.
        # Also zum Beispiel die Vektoren von 'Frau' und 'König' addieren, und danach 'Mann' abziehen.
        # Danach wird das Wort des passendsten Vektor ausgegeben. Klappt natürlich nur mit Vektoren, die 
        # im Text vorkommen, und ist erst ab einer gewissen Korpusgrösse sinnvoll.

        result = self.model.most_similar_cosmul(positive = keyword_list_1, negative = keyword_list_2)
        
        return result
    
    def compute_similarity(self, keyword_1, keyword_2):
        # Auch die Ähnlichkeit zweier Vektoren kann man berechnen. 
        similarity = self.model.similarity(keyword_1, keyword_2)
        return similarity
        
    def visualize_nearest_neighbours(self, dimensions, keyword_list, topn):
        pca = PCA(n_components=2 if dimensions == '2d' else 3)
        fig = go.Figure()
        
        all_words = set(keyword_list)  # Vermeide Duplikate
        all_vectors = []
        labels = []
    
        # Sammle alle Vektoren und Labels
        for word in keyword_list:
            all_vectors.append(self.model[word])  # Das ausgewählte Wort
            labels.append(word)  # Label für das ausgewählte Wort
            neighbors = self.model.most_similar(word, topn=topn)
            for neighbor, _ in neighbors:
                all_vectors.append(self.model[neighbor])
                labels.append(neighbor)  # Label für den Nachbarn
                all_words.add(neighbor)
        
        # Berechne PCA für alle Vektoren
        all_vectors = pca.fit_transform(np.array(all_vectors))
        word_to_pca = {word: all_vectors[i] for i, word in enumerate(all_words)}
    
        # Trace für die ausgewählten Wörter
        selected_word_coords = np.array([word_to_pca[word] for word in keyword_list])
        selected_word_labels = [word for word in keyword_list]
        if dimensions == '2d':
            fig.add_trace(go.Scatter(x=selected_word_coords[:, 0], y=selected_word_coords[:, 1],
                                     mode='markers+text', text=selected_word_labels,
                                     marker=dict(color='red', size=10), name='Selected Words',
                                     textposition='top center', textfont=dict(size=16)))
        else:
            fig.add_trace(go.Scatter3d(x=selected_word_coords[:, 0], y=selected_word_coords[:, 1], z=selected_word_coords[:, 2],
                                       mode='markers+text', text=selected_word_labels,
                                       marker=dict(color='red', size=10), name='Selected Words',
                                       textposition='top center', textfont=dict(size=16)))
    
        # Trace für die Nachbarn
        for word in keyword_list:
            neighbors = self.model.most_similar(word, topn=topn)
            neighbor_coords = np.array([word_to_pca[neighbor[0]] for neighbor in neighbors])
            neighbor_labels = [neighbor[0] for neighbor in neighbors]
            if dimensions == '2d':
                fig.add_trace(go.Scatter(x=neighbor_coords[:, 0], y=neighbor_coords[:, 1],
                                         mode='markers+text', text=neighbor_labels,
                                         marker=dict(size=8), name=f'Neighbors of {word}',
                                         textposition='bottom center', textfont=dict(size=12)))
            else:
                fig.add_trace(go.Scatter3d(x=neighbor_coords[:, 0], y=neighbor_coords[:, 1], z=neighbor_coords[:, 2],
                                           mode='markers+text', text=neighbor_labels,
                                           marker=dict(size=8), name=f'Neighbors of {word}',
                                           textposition='bottom center', textfont=dict(size=12)))
    
        # Anpassen des Layouts
        fig.update_layout(title='Selected Words and Their Neighbors')
        return fig
    
    ######################

    def prepare_files_for_tensorflow_visualization(self):
        outf_vecs = self.modelpath.replace(".word_vectors", "_vectors.tsv")
        outf_meta = self.modelpath.replace(".word_vectors", "_metadata.tsv")
        with open(self.modelpath, "r", encoding='utf-8') as vec_file:
            with open(outf_vecs, "w", encoding='utf-8') as vec_tsv:
                with open(outf_meta, "w", encoding='utf-8') as meta_tsv:
                    tsv_writer1 = csv.writer(vec_tsv, delimiter='\t')
                    tsv_writer2 = csv.writer(meta_tsv, delimiter='\t')
                    # erste Linie überspringen, sie ist ohne relevanten Inhalt
                    line = vec_file.readline()
                    line = vec_file.readline()
                    currvec = line.strip().split(' ')
                    counter = 0
                    while line:
                        counter += 1
                        if currvec[0] == "":
                            break
                        tsv_writer2.writerow([currvec[0]])
                        currvec.pop(0)
                        tsv_writer1.writerow(currvec)
                        currvec = vec_file.readline().strip().split(' ')
                    print("Done!")