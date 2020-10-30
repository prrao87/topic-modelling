# Building a Scalable Topic Model Workflow

## Latent Dirichlet Allocation
Topic modelling is an unsupervised machine learning technique to discover the main ‘topics’, or themes,in a collection of unstructured documents. A ‘topic’ here refers to a cluster of words that represents a larger concept from the real world. Each document in a corpus can be imagined as consisting of multiple topics in different proportions all at once — for example, in an article about a major airline procuring new aircraft, it is reasonable to expect many words related to finance, geopolitics, travel policy, as well as passenger trends and market events that led to the deal taking place. A document can thus be composed of several topics, each consisting of specific words (that may or may not overlap between topics).

Topic modelling encapsulates these ideas into a mathematical framework that discovers clusters of word distributions representing overall themes within the corpus, making it a useful technique to analyze very large datasets for their content.The mathematical goal of topic modelling is to fit a model’s parameters to the given data using heuristic rules, such that there is a maximum likelihood that the data arose from the model. Such methods are known as parametric methods, among which ​Latent Dirichlet Allocation​ (LDA) is by far the most popular.

Several Python-based implementations of LDA exist - the focus in this repo is to study the topic modelling results of two specific implementations: __Gensim__ and __PySpark__. Broadly speaking, both methods perform similar steps, but it is entirely up to the user to preprocess the text as required beforehand. Specifically for news articles and communication research corpora, the below sequence of preprocessing steps (as per [[1]](#references)) are found to provide good topic model results downstream:

1. Sentence detection
1. Tokenization
1. Lowercasing
1. Normalizing (cleaning unwanted symbols and artifacts)
1. Stopword removal
1. Lemmatization
1. Topic model training

### Stopword removal
Because a large proportion of text in any corpus contains repetitive and meaningless words (with respect to interpreting topics) such as "a", "and", or "the", as well as a host of other common verbs/nouns, stopword removal is a very important step in text preprocessing. A stopword list is **highly domain-specific**, so careful thought should be put in beforehand to curate a reasonable list of stopwords. Topic modelling is typically an iterative process, where multiple training runs of the model are done to identify more stopwords that hinder topic interpretation.

### Why perform lemmatization?
Although lemmatization is not strictly necessary, Maier et. al. [[1]](#references) state in their review paper of topic modelling literature that following a clear sequence of steps during text preprocessing can yield better and more interpretable topics. Since lemmatization reduces words to their root form, it results in large-scale feature reduction by combining words that are very similar to each other (such as *kill*, *killed* and *killing*).


## Set up Python Environment

First, set up virtual environment and install the required libraries from requirements.txt:

```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

For further development, simply activate the existing virtual environment.

```
source venv/bin/activate
```

#### Language model for spaCy

For NLP-related tasks such as lemmatization and stopword lists, we use the SpaCy library's "small" English language model (the "large" one can be used as well, but this can take longer to load and generate results). Download the spaCy language model as follows:

```
python3 -m spacy download en_core_web_sm
```

## Set up PySpark Environment

LDA is done in PySpark using Spark's ML pipeline as well as the external [__SparkNLP__](https://nlp.johnsnowlabs.com/) library (for lemmatization).

To set up PySpark, first [install Java 8+ (using OpenJDK)](https://openjdk.java.net/install/).

#### Install PySpark and SparkNLP 2.4+
```
pip install pyspark==2.4.4 spark-nlp==2.4.5
```
Specify the PySpark location as an environment variable:

```
export SPARK_HOME=/path/to/spark/
export PYSPARK_PYTHON=python3
export PYSPARK_DRIVER_PYTHON=python3
```

## Obtain Raw Data
As an example, the New York Times dataset from Kaggle is used (download instructions are in the folder `data/`). This is an English-language news dataset of 8,800 articles from the New York Times over a few months in 2016. The structure of the dataset, once preprocessed, is as follows:

| date | url | content |
|:------:| :-----: | :-------: |
| 2016-06-30 |  http://www.nytimes.com/2016/06/30/sports/baseb..   | WASHINGTON — Stellar pitching kept the Mets af...  |
| 2016-06-30 |  http://www.nytimes.com/2016/06/30/nyregion/may..   | Mayor Bill de Blasio’s counsel and chief legal...
| ...|  ... |  ... |

The dataset contains article content from a raw HTML dump, so it is full of unnecessary symbols and artifacts.

## Create initial stopword list
To train a topic model, a hand-curated, domain-specific stopword list is necessary. Run the script `topic_model/stopwords/create_stopword_list.py`.
```
cd topic_model/stopwords
python3 create_stopword_list.py
```
This script pulls the default spaCy stopword list, and adds a number of news article-specific vocabulary to the stopword list (obtained after some trial and error and inspecting initial model results).

## Train topic model
See the [src](https://github.com/prrao87/topic-modelling/tree/master/src) directory.

---

## References

[1] Maier, D., Waldherr, A., Miltner, P., Wiedemann, G., Niekler, A., Keinert, A., ... Adam, S. (2018). Applying LDA topic modeling in communication research: Toward a valid and reliable methodology. *Communication Methods and Measures*, 12(2–3), 93–118. doi:10.1080/19312458.2018.1430754 [Taylor & Francis Online](https://www.tandfonline.com/servlet/linkout?suffix=CIT0040&dbid=20&doi=10.1080%2F19312458.2018.1458084&key=10.1080%2F19312458.2018.1430754&tollfreelink=2_18_091d52e2c25fb605f624551cc29e5f412ee28f10d2308cd98d03acb52762af29).
