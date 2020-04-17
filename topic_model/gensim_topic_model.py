import re
import plac
import pandas as pd
import spacy
from typing import List
from math import sqrt, ceil
# gensim
from gensim import corpora
from gensim.models.ldamulticore import LdaMulticore
# plotting
from matplotlib import pyplot as plt
from wordcloud import WordCloud
# progress bars
from tqdm import tqdm
tqdm.pandas()


# ============  Methods  =================s
def read_data(filepath: str) -> pd.DataFrame:
    "Read in a tab-separated file with date, headline and news content"
    df = pd.read_csv(filepath, sep='\t', header=None,
                     names=['date', 'headline', 'content'])
    df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    "Extract relevant text from DataFrame using a regex"
    # Regex pattern for only alphanumeric, hyphenated text with 3 or more chars
    pattern = re.compile(r"[A-Za-z0-9\-]{3,50}")
    df['clean'] = df['content'].str.findall(pattern).str.join(' ')
    return df


def lemmatize(text: str, nlp, stopwords: List[str]) -> List[str]:
    "Perform lemmatization and stopword removal in the clean text"
    doc = nlp(text)
    lemma_list = [str(tok.lemma_).lower() for tok in doc
                  if tok.is_alpha and tok.text.lower() not in stopwords]
    return lemma_list


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    "Preprocess text in each row of the DataFrame"
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    stopwords = nlp.Defaults.stop_words  # Default spaCy stopword list
    df['lemmas'] = df['clean'].progress_apply(lambda row: lemmatize(row, nlp, stopwords))
    return df.drop('clean', axis=1)


def run_lda_multicore(text_df, num_topics, iterations=200, epochs=20, workers=7):
    """Run Gensim's multicore LDA topic modelling algorithm
       Choose number of workers for multicore LDA as (num_physical_cores - 1)
    """
    print("Running LDA model...")
    id2word = corpora.Dictionary(text_df['lemmas'])
    # Filter out words that occur in less than 2% documents or more than 50% of the documents.
    id2word.filter_extremes(no_below=0.02, no_above=0.8)
    corpus = [id2word.doc2bow(text) for text in text_df['lemmas']]
    # LDA Model
    lda_model = LdaMulticore(
        corpus=corpus,
        id2word=id2word,
        workers=workers,
        num_topics=num_topics,
        random_state=1,
        chunksize=2048,
        passes=epochs,
        iterations=iterations,
    )
    print("Finished!")
    return lda_model, corpus


def plot_wordclouds(topics):
    cloud = WordCloud(
        background_color='white',
        width=1000,
        height=800,
        colormap='cividis',
    )
    # Define the size of the subplot matrix as a function of num_topics
    dim = ceil(sqrt(len(topics)))
    fig = plt.figure(figsize=(15, 15))

    for i in range(len(topics)):
        print("Topic {}".format(i + 1))
        ax = fig.add_subplot(dim, dim, i + 1) 
        topic_words = dict(topics[i][1])
        wordcloud = cloud.generate_from_frequencies(topic_words)
        ax.imshow(wordcloud)
        ax.set_title('Topic {}'.format(i + 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(length=0)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.margins(x=0, y=0)
    plt.tight_layout()
    fig.savefig("gensim-topics.png")


@plac.annotations(
    num_topics=("Number of topics in LDA", "option", "n", int),
    iterations=("Iterations in LDA", "option", "i", int),
    epochs=("Training epochs", "option", "e", int)
)
def main(num_topics=15, iterations=200, epochs=20):
    df = read_data("../data/nytimes.tsv")
    df_clean = clean_data(df)
    df_preproc = preprocess(df_clean)
    model, corpus = run_lda_multicore(df_preproc, num_topics=num_topics,
                                      iterations=iterations, epochs=epochs)
    plot_wordclouds(model.show_topics(formatted=False, num_topics=num_topics, num_words=15))


if __name__ == "__main__":
    plac.call(main)