# Models

## Train a Topic Model in Gensim
The Gensim LDA module takes in a training corpus and allows us to perform several NLP preprocessing steps, such as tokenization and stopword removal. For consistency with the upcoming PySpark modelling approach, Pandas is used to store the document corpus in a DataFrame, following which spaCy is used to perform tokenization and lemmatization. The [multicore version](https://radimrehurek.com/gensim/models/ldamulticore.html#module-gensim.models.ldamulticore) of Gensim's LDA model is used in this exercise to speed up the training loop.

The Gensim topic model script accepts command line arguments as follows.

```sh
python3 topic_model_gensim.py --topics 20 --iter 200 --epochs 20 --minDF 0.02 --maxDF 0.8
```

Obtain a description of the command line arguments by typing the `-h` command.

```sh
python3 topic_model_gensim.py -h
usage: topic_model_gensim.py [-h] [--topics TOPICS] [--iter ITER]
                             [--epochs EPOCHS] [--minDF MINDF] [--maxDF MAXDF]
                             [--n_proc N_PROC]

optional arguments:
  -h, --help            show this help message and exit
  --topics TOPICS, -t TOPICS
                        Number of topics in LDA
  --iter ITER, -i ITER  Max iterations in LDA
  --epochs EPOCHS, -e EPOCHS
                        Max number of epochs for Gensim
  --minDF MINDF, -m1 MINDF
                        Minimum document frequency
  --maxDF MAXDF, -m2 MAXDF
                        Maximum document frequency
  --n_proc N_PROC, -n N_PROC
                        Number of CPU processes

```

---

## Train a Topic Model in PySpark
The LDA topic model pipeline in PySpark uses its DataFrame API. The NYT tabular data is read into a Spark DataFrame, following which a parallelized approach is applied to clean, lemmatize and run the topic model pipeline. For consistency, the same regular expression syntax is used in both Gensim and PySpark (to provide similar input text) to the model. In addition, similar hyperparameters are used as far as possible during the training process.

The PySpark topic model script **requires a manual specification** of the external Spark NLP library (which is used for large-scale, parallelized lemmatization of terms in the data), followed by the optional arguments for the hyperparameters.

```sh
spark-submit --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.0 topic_model_pyspark.py --topics 20 --iter 200 --vocabsize 5000 --minDF 0.02 --maxDF 0.8
```

Obtain a description of the command line arguments by typing the `-h` command.
```sh
spark-submit --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.0 topic_model_pyspark.py --help
usage: topic_model_pyspark.py [-h] [--topics TOPICS] [--iter ITER]
                              [--vocabsize VOCABSIZE] [--minDF MINDF]
                              [--maxDF MAXDF]

optional arguments:
  -h, --help            show this help message and exit
  --topics TOPICS, -t TOPICS
                        Number of topics in LDA
  --iter ITER, -i ITER  Max iterations in LDA
  --vocabsize VOCABSIZE, -v VOCABSIZE
                        Max vocabSize in LDA
  --minDF MINDF, -m1 MINDF
                        Minimum document frequency
  --maxDF MAXDF, -m2 MAXDF
                        Maximum document frequency

```

---
