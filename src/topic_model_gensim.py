import re
import argparse
import pandas as pd
import spacy
from spacy.tokens.doc import Doc
from typing import List, Dict, Set, Any
from math import ceil
# Concurrency
from joblib import Parallel, delayed
from functools import partial
from multiprocessing import cpu_count
# gensim
from gensim import corpora
from gensim.models.ldamulticore import LdaMulticore
# plotting
from matplotlib import pyplot as plt
from wordcloud import WordCloud

inputfile = "../data/nytimes.tsv"
stopwordfile = "stopwords/custom_stopwords.txt"
# Load spaCy model
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner', 'tagger'])
nlp.add_pipe(nlp.create_pipe('sentencizer'))


# ============  Methods  =================s
def read_data(filepath: str) -> pd.DataFrame:
    "Read in a tab-separated file with date, headline and news content"
    df = pd.read_csv(filepath, sep='\t', header=None,
                     names=['date', 'headline', 'content'])
    df['date'] = pd.to_datetime(df['date'], format="%Y-%m-%d")
    return df


def get_stopwords(stopwordfile: str) -> Set[str]:
    "Read in stopwords"
    with open(stopwordfile) as f:
        stopwords = []
        for line in f:
            stopwords.append(line.strip("\n"))
    return set(stopwords)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    "Extract relevant text from DataFrame using a regex"
    # Regex pattern for only alphanumeric, hyphenated text with 3 or more chars
    pattern = re.compile(r"[A-Za-z0-9\-]{3,50}")
    df['clean'] = df['content'].str.findall(pattern).str.join(' ')
    return df


def lemmatize(doc: Doc, stopwords: List[str]) -> List[str]:
    "Perform lemmatization and stopword removal in the clean text"
    lemma_list = [str(tok.lemma_).lower() for tok in doc
                  if tok.is_alpha and tok.text.lower() not in stopwords]
    return lemma_list


def chunker(iterable: List[str], total_length: int, chunksize: int) -> List[str]:
    return (iterable[pos: pos + chunksize] for pos in range(0, total_length, chunksize))


def flatten(list_of_lists: List[List[str]]) -> List[str]:
    "Flatten a list of lists to a combined list"
    return [item for sublist in list_of_lists for item in sublist]


def process_chunk(stopwords: Set[str], texts: List[str]) -> List[str]:
    preproc_pipe = []
    for doc in nlp.pipe(texts, batch_size=20):
        preproc_pipe.append(lemmatize(doc, stopwords))
    return preproc_pipe


def preprocess_concurrent(texts: List[str], stopwords: Set[str], chunksize: int=100):
    executor = Parallel(n_jobs=params['n_proc'], backend='multiprocessing', prefer="processes")
    do = delayed(partial(process_chunk, stopwords))
    tasks = (do(chunk) for chunk in chunker(texts, len(texts), chunksize=chunksize))
    result = executor(tasks)
    return flatten(result)


def run_lda_multicore(
        text_df: pd.DataFrame,
        params: Dict[str, Any],
        workers: int = 7
    ):
    """Run Gensim's multicore LDA topic modelling algorithm
       Choose number of workers for multicore LDA as (num_physical_cores - 1)
    """
    print("Applying LDA algorithm...")
    id2word = corpora.Dictionary(text_df['lemmas'])
    # Filter out words that occur in less than 2% documents or more than 50% of the documents.
    id2word.filter_extremes(no_below=params['minDF'], no_above=params['maxDF'])
    corpus = [id2word.doc2bow(text) for text in text_df['lemmas']]
    # LDA Model
    lda_model = LdaMulticore(
        corpus=corpus,
        id2word=id2word,
        workers=workers,
        num_topics=params['topics'],
        random_state=1,
        chunksize=2048,
        passes=params['epochs'],
        iterations=params['iter'],
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

    for idx, word_weights in enumerate(topics):
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


def main(params: Dict[str, Any]) -> None:
    df = read_data(inputfile)
    stopwords = get_stopwords(stopwordfile)
    print(f'Beginning text preprocessing...')
    df_preproc = clean_data(df)
    df_preproc['lemmas'] = preprocess_concurrent(df_preproc['clean'], stopwords)
    print('Finished preprocessing {} samples'.format(df_preproc.shape[0]))
    model, corpus = run_lda_multicore(df_preproc, params, workers=params['n_proc'])
    topic_list = model.show_topics(formatted=False,
                                   num_topics=params['topics'],
                                   num_words=20)
    # Store topic words amd weights as a list of dicts
    topics = [dict(item[1]) for item in topic_list]
    plot_wordclouds(topics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topics", "-t", type=int, default=20, help="Number of topics in LDA")
    parser.add_argument("--iter", "-i", type=int, default=200, help="Max iterations in LDA")
    parser.add_argument("--epochs", "-e", type=int, default=20, help="Max number of epochs for Gensim")
    parser.add_argument("--minDF", "-m1", type=float, default=0.02, help="Minimum document frequency")
    parser.add_argument("--maxDF", "-m2", type=float, default=0.8, help="Maximum document frequency")
    parser.add_argument("--n_proc", "-n", type=int, default=cpu_count() + 1, help="Number of CPU processes")

    params = vars(parser.parse_args())
    # Run LDA
    main(params)
