import re
import plac
import pandas as pd
import spacy
from typing import List, Dict, Any
from math import ceil
# gensim
from gensim import corpora
from gensim.models.ldamulticore import LdaMulticore
# plotting
from matplotlib import pyplot as plt
from wordcloud import WordCloud
# progress bars
from tqdm import tqdm
tqdm.pandas()


inputfile = "../data/nytimes.tsv"
stopwordfile = "stopwords/custom_stopwords.txt"


# ============  Methods  =================s
def read_data(filepath: str) -> pd.DataFrame:
    "Read in a tab-separated file with date, headline and news content"
    df = pd.read_csv(filepath, sep='\t', header=None,
                     names=['date', 'headline', 'content'])
    df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")
    return df


def get_stopwords(stopwordfile: str) -> List[str]:
    "Read in stopwords"
    with open(stopwordfile) as f:
        stopwords = []
        for line in f:
            stopwords.append(line.strip("\n"))
    return stopwords


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


def preprocess(df: pd.DataFrame, stopwords: List[str]) -> pd.DataFrame:
    "Preprocess text in each row of the DataFrame"
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    df['lemmas'] = df['clean'].progress_apply(lambda row: lemmatize(row, nlp, stopwords))
    return df.drop('clean', axis=1)


def run_lda_multicore(text_df: pd.DataFrame, params: Dict[str, Any], workers: int=7):
    """Run Gensim's multicore LDA topic modelling algorithm
       Choose number of workers for multicore LDA as (num_physical_cores - 1)
    """
    print("Running LDA model...")
    id2word = corpora.Dictionary(text_df['lemmas'])
    # Filter out words that occur in less than 2% documents or more than 50% of the documents.
    id2word.filter_extremes(no_below=params['minDF'], no_above=params['maxDF'])
    corpus = [id2word.doc2bow(text) for text in text_df['lemmas']]
    # LDA Model
    lda_model = LdaMulticore(
        corpus=corpus,
        id2word=id2word,
        workers=workers,
        num_topics=params['num_topics'],
        random_state=1,
        chunksize=2048,
        passes=params['epochs'],
        iterations=params['iterations'],
    )
    return lda_model, corpus


def plot_wordclouds(topics: List[Dict[str, float]],
                    colormap: str="cividis") -> None:
    cloud = WordCloud(
        background_color='white',
        width=600,
        height=400,
        colormap=colormap,
        prefer_horizontal=1.0,
    )

    num_topics = len(topics)
    # Hacky way to scale figure size based on the number of topics specified
    fig_width = min(ceil(0.6 * num_topics + 6), 20)
    fig_height = min(ceil(0.65 * num_topics), 20)
    fig = plt.figure(figsize=(fig_width, fig_height))

    for idx, word_weights in tqdm(enumerate(topics), total=num_topics):
        ax = fig.add_subplot((num_topics / 5) + 1, 5, idx + 1)
        wordcloud = cloud.generate_from_frequencies(word_weights)
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.set_title('Topic {}'.format(idx + 1))
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(length=0)

    plt.tick_params(labelsize=14)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.margins(x=0.1, y=0.1)
    st = fig.suptitle("LDA Topics", y=0.92)
    fig.savefig("gensim-topics.png", bbox_extra_artists=[st], bbox_inches='tight')


@plac.annotations(
    num_topics=("Number of topics in LDA", "option", "n", int),
    iterations=("Iterations in LDA", "option", "i", int),
    epochs=("Training epochs", "option", "e", int),
    minDF=("Minimum document frequency for LDA", "option", "m1", float),
    maxDF=("Maximum document frequency for LDA", "option", "m2", float)
)
def main(num_topics=15, iterations=200, epochs=20, minDF=0.02, maxDF=0.8) -> None:
    params = {
        'num_topics': num_topics,
        'iterations': iterations,
        'epochs': epochs,
        'minDF': minDF,
        'maxDF': maxDF,
    }
    df = read_data(inputfile)
    stopwords = get_stopwords(stopwordfile)
    df_clean = clean_data(df)
    df_preproc = preprocess(df_clean, stopwords)
    model, corpus = run_lda_multicore(df_preproc, params)
    topic_list = model.show_topics(formatted=False,
                                   num_topics=params['num_topics'],
                                   num_words=15)
    # Store topic words amd weights as a list of dicts
    topics = [dict(item[1]) for item in topic_list]
    plot_wordclouds(topics)


if __name__ == "__main__":
    plac.call(main)
