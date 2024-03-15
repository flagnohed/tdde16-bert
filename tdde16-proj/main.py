import pandas as pd
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP
import hdbscan
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
from gensim.models import LdaModel
from gensim.corpora.dictionary import Dictionary
import preprocess as pp

def calculate_coherence(model, topics, speeches):
    # this function is blatantly stolen from:
    # https://www.theanalyticslab.nl/topic-modeling-with-bertopic/

    documents = pd.DataFrame({"Document": speeches,
                            "ID": range(len(speeches)),
                            "Topic": topics})
    documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
    cleaned_docs = model._preprocess_text(documents_per_topic.Document.values)

    vectorizer = model.vectorizer_model
    analyzer = vectorizer.build_analyzer()
   
    words = vectorizer.get_feature_names_out()
    tokens = [analyzer(doc) for doc in cleaned_docs]
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    topic_words = [[words for words, _ in model.get_topic(topic)] 
                for topic in range(len(set(topics))-1)]

    coherence_model = CoherenceModel(topics=topic_words, 
                                 texts=tokens, 
                                 corpus=corpus,
                                 dictionary=dictionary, 
                                 coherence='u_mass')

    s = coherence_model.get_coherence_per_topic()
    return coherence_model.get_coherence()

def get_embeddings(df):
    # too slow, took 12 hrs per run
    model = SentenceTransformer('KBLab/sentence-bert-swedish-cased')
    speeches = df.speech.to_list()
    embeddings = model.encode(speeches, show_progress_bar=True)
    return embeddings

def main():
    df = pd.read_json("sp.json.zip", orient='records')

    umap = UMAP(low_memory=False, metric='cosine', min_dist=0.0)
    hdb = hdbscan.HDBSCAN()

    embedding = 'paraphrase-multilingual-MiniLM-L12-v2'
    model = BERTopic(embedding_model=embedding, 
                    language='swedish',
                    verbose=True,
                    vectorizer_model=CountVectorizer(),
                    umap_model=umap,
                    hdbscan_model=hdb)

    years = df.year.to_list()
    timestamps = df.date.to_list()
    speeches = df.speech.to_list()
    topics, probabilities = model.fit_transform(speeches)
    
    topic_df = pd.DataFrame({"year": years, "date":timestamps, 
                             "speech":speeches, "topic": topics})
    
    model.save("models", serialization="safetensors", 
               save_ctfidf=True, save_embedding_model=embedding)
    

    topic_df.to_json("topic_df.json")


def get_graphs_bert():
    df = pd.read_json("topic_df.json")

    model = BERTopic.load("models",)
    """
    for y in set(df["year"]):
        temp_df = df.loc[df["year"] == y]
        print(temp_df.head())
        tot = model.topics_over_time(temp_df.speech.to_list(), temp_df.date.to_list(), temp_df.topic.to_list())

        fig = model.visualize_topics_over_time(tot, top_n_topics=6)
        
        fig.write_html(y[:4]+"_over_time_3.html")
    """
    topics = model.get_topic_info().head(20)
    print(topics)
    print("coherence: ", calculate_coherence(model, df["topic"].to_list(), df["speech"].to_list()))
    print(model.get_topics())
    print("================")
    print('Total number of topics: ', len(set(model.topics_)))
    print('Total number of outliers: ', model.get_topic_freq(-1))


def train_lda():
    num_topics = 6
    for d in pp.DATASETS:
        speeches = pp.preprocess_zip_lda(d)    
        dictionary = Dictionary(speeches)
        corpus = [dictionary.doc2bow(doc) for doc in speeches]
        lda = LdaModel(num_topics=num_topics, passes=1, corpus=corpus)

        # get coherence both umass and c_v
        # will do one for each dataset and then take the average.
        # this is because we dont have access to dates.
        coherence_lda = CoherenceModel(
            model = lda, texts=speeches, dictionary=dictionary,
            coherence='u_mass'
        )
        print("U_MASS: ", coherence_lda.get_coherence())
        coherence_lda = CoherenceModel(
            model = lda, texts=speeches, dictionary=dictionary,
            coherence='c_v'
        )
        print("C_V: ", coherence_lda.get_coherence())


if __name__ == "__main__":
    # main()
    get_graphs_bert()
    # train_lda()
   





