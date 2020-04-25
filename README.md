# Building a Scalable Topic Model Workflow

## Latent Dirichlet Allocation
By far the most popular method to model themes and topics in discrete document corpora, Latent Dirichlet Allocation (LDA) is a generative probabilistic model that represents documents as a distribution over words belonging to a topic. In general, topic modelling is an *unsupervised modelling technique* that requires human inspection as well as some level of domain expertise to label the extracted distribution of words as a "topic". A key feature of LDA is that it is a "multiple membership" mixture model, meaning that the same words can appear multiple times in different topics. 

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

## Train a Topic Model in Gensim
The Gensim LDA module takes in a training corpus and allows us to perform several NLP preprocessing steps, such as tokenization and stopword removal. For consistency with the upcoming PySpark modelling approach, Pandas is used to store the document corpus in a DataFrame, following which spaCy is used to perform tokenization and lemmatization. The [multicore version](https://radimrehurek.com/gensim/models/ldamulticore.html#module-gensim.models.ldamulticore) of Gensim's LDA model is used in this exercise to speed up the training loop.

The Gensim topic model script accepts command line arguments as follows.

```
cd topic_model
python topic_model_gensim.py --num-topics 15 --iterations 200 --epochs 20
```

Obtain a description of the command line arguments by typing the `-h` command.

```
python topic_model_gensim.py -h
usage: topic_model_gensim.py [-h] [-n 15] [-i 200] [-e 20] [-m1 0.02]
                             [-m2 0.8]

optional arguments:
  -h, --help            show this help message and exit
  -n 15, --num-topics 15
                        Number of topics in LDA
  -i 200, --iterations 200
                        Iterations in LDA
  -e 20, --epochs 20    Training epochs
  -m1 0.02, --minDF 0.02
                        Minimum document frequency for LDA
  -m2 0.8, --maxDF 0.8  Maximum document frequency for LDA
```

## Train a Topic Model in PySpark
The LDA topic model pipeline in PySpark uses its DataFrame API. The NYT tabular data is read into a Spark DataFrame, following which a parallelized approach is applied to clean, lemmatize and run the topic model pipeline. For consistency, the same regular expression syntax is used in both Gensim and PySpark (to provide similar input text) to the model. In addition, similar hyperparameters are used as far as possible during the training process.

The PySpark topic model script accepts command line arguments, as well as a manual specification of the external Spark NLP library (for large-scale, parallelized lemmatization).

```
cd topic_model
spark-submit --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.0 topic_model_pyspark.py 15 100 5000 0.02 0.8
```
The arguments provided to the Python script through `sys.argv` are described below:
```
spark-submit --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.0 topic_model_pyspark.py
        15     # Number of topics in LDA
        100    # Max. iterations in LDA
        5000   # Max. vocabulary size to consider in LDA
        0.02   # Min. document frequency in LDA
        0.8    # Max. document frequency in LDA
```

## References

[1] Maier, D., Waldherr, A., Miltner, P., Wiedemann, G., Niekler, A., Keinert, A., ... Adam, S. (2018). Applying LDA topic modeling in communication research: Toward a valid and reliable methodology. *Communication Methods and Measures*, 12(2–3), 93–118. doi:10.1080/19312458.2018.1430754 [Taylor & Francis Online](https://www.tandfonline.com/servlet/linkout?suffix=CIT0040&dbid=20&doi=10.1080%2F19312458.2018.1458084&key=10.1080%2F19312458.2018.1430754&tollfreelink=2_18_091d52e2c25fb605f624551cc29e5f412ee28f10d2308cd98d03acb52762af29)


