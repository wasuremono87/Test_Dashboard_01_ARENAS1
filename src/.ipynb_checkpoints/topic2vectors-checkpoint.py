from top2vec import Top2Vec
import re
import plotly.express as px
import pandas as pd


class Topic2vectors():

    def __init__(self, modelpath="../data/top2vec/testmodel.model"):
        self.modelpath = modelpath
        print(modelpath)
        

    def make_model(self, all_documents, embedding_model="doc2vec"): # all_documents = Liste mit Listen von Tokens (--> df["Texts_Tokenized_ALL"] oder df["Texts_Lemmatized_ALL"] 
        all_documents = [" ".join(doc) for doc in all_documents if doc]

        # TODO passe die Parameter an (siehe README oder Dokumentation)
        model = Top2Vec(
            all_documents, embedding_model="doc2vec", speed="deep-learn", workers=4, min_count=6
        )
        
        # TODO hier den Dateipfad zum Modell angeben. Format: .model
        model.save(self.modelpath)

    def top2vec_overview(self, output_file="../data/top2vec/overview_model.csv"):
        # pass den Dateipfad an, um dein erstelltes Modell zu laden
        model = Top2Vec.load(self.modelpath)
    
        # .get_topic_sizes() gibt eine Liste mit der Grösse der Topics im Korpus zurück, und eine mit den id's der Topics.
        topic_sizes, topic_nums = model.get_topic_sizes()
    
        """
        .get_topics() nimmt die Anzahl Topics des Modelles als Argument und gibt drei Listen zurück:
            1. eine Liste mit einer Liste pro Topic mit den Top 50 Worten, die das Topic ausmachen
            2. eine Liste mit einer Liste pro Topic mit der Cosinus Distanz jedes Wortes zum entsprechenden Topic-Embedding
            3. eine Liste mit den IDs der Topics
        """
        topic_words, word_scores, topic_nums = model.get_topics(model.get_num_topics())
    
        with open(output_file, "w", encoding="utf-8") as csv_out:
            csv_out.write("topic_number,words_in_topic,number_of_documents\n")
    
            for topic_num in topic_nums:
                string_topic_words = re.sub(r"[\n,\[,\]]", " ", str(topic_words[topic_num]))
                #print(string_topic_words)
                csv_out.write(
                    f"{topic_num},{string_topic_words},{topic_sizes[topic_num]}\n"
                )
    
        """
        Visualisierung:
            Balkendiagramm aus den Topics und ihrer Anzahl. 
            Anpassbar sind  der Titel und die Beschriftungen der Achsen in .update_layout,
                            die Groesse des Diagramms in .update_layout,
                            die Breite der Balken in .update_traces,
                            die Anzahl angezeigter Worte pro Topic, momentan 7.
        """
        topic_df = pd.read_csv(csv_path)
        
        # TODO passe die Anzahl der angezeigten Worte pro Topic an, momentan 7, falls gewünscht
        shortened_words = topic_df["words_in_topic"].str.split().str[:7].apply(" ".join)
        short_words_with_ind = []
        for ind, words in enumerate(shortened_words):
            new_words = words + " :" + str(ind)
            short_words_with_ind.append(new_words)
    
        fig = px.bar(
            topic_df,
            x=topic_df["number_of_documents"],
            y=short_words_with_ind,
            orientation="h",
            color="number_of_documents",
        )
    
        # TODO passe den Titel und evtl. die Achsenbeschriftungen an
        fig.update_layout(
            title="Balkendiagramm: Beispieltitel",
            xaxis_title="Topics",
            yaxis_title="Anzahl Dokumente",
            yaxis={"categoryorder": "total ascending"},
        )
        # TODO passe die Breite der Balken und die Groesse des Diagramms an, falls gewünscht
        fig.update_traces(width=0.85)
        fig.update_layout(width=1400, height=1200)
        fig.show()


    def top2vec_inspect_docs(self, topic_of_interest=0, output_file="../data/top2vec/inspect_model.txt"):
        #  Dateipfad zum Modell angeben 
        model = Top2Vec.load(self.modelpath)
        topic_sizes, topic_nums = model.get_topic_sizes()
        
        # TODO die ID des Topic of Interest eingeben, achtung, indexing beginnt bei 0. ablesbar in der Visualisierung
        number_of_docs_in_topic = topic_sizes[topic_of_interest]
        
        # TODO Hier die Anzahl Dokumente angeben, die angezeigt werden soll. Default sind alle
        document_count = number_of_docs_in_topic
        print(document_count)
        
        documents, doc_scores, doc_ids = model.search_documents_by_topic(topic_of_interest, document_count)
        
        # TODO Dateipfad zum gewuenschten Outputfile angeben
        with open(output_file, 'w', encoding='utf-8') as outfile:
            outfile.write(str(documents[topic_of_interest]))
            for doc in documents:
                outfile.write(doc)