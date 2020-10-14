# Models

## Train a Topic Model in Gensim
The Gensim LDA module takes in a training corpus and allows us to perform several NLP preprocessing steps, such as tokenization and stopword removal. For consistency with the upcoming PySpark modelling approach, Pandas is used to store the document corpus in a DataFrame, following which spaCy is used to perform tokenization and lemmatization. The [multicore version](https://radimrehurek.com/gensim/models/ldamulticore.html#module-gensim.models.ldamulticore) of Gensim's LDA model is used in this exercise to speed up the training loop.

The Gensim topic model script accepts command line arguments as follows.

```
cd topic_model
python topic_model_gensim.py --num-topics 15 --iterations 200 --epochs 20 --minDF 0.02 --maxDF 0.8
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

---

## Train a Topic Model in PySpark
The LDA topic model pipeline in PySpark uses its DataFrame API. The NYT tabular data is read into a Spark DataFrame, following which a parallelized approach is applied to clean, lemmatize and run the topic model pipeline. For consistency, the same regular expression syntax is used in both Gensim and PySpark (to provide similar input text) to the model. In addition, similar hyperparameters are used as far as possible during the training process.

The PySpark topic model script accepts command line arguments, as well as a manual specification of the external Spark NLP library (for large-scale, parallelized lemmatization).

```
cd topic_model
spark-submit --packages com.johnsnowlabs.nlp:spark-nlp_2.11:2.4.0 topic_model_pyspark.py --num-topics 15 --iterations 200 --vocabsize 5000 --minDF 0.02 --maxDF 0.8
```

Obtain a description of the command line arguments by typing the `-h` command.
```
spark-submit topic_model_pyspark --help
usage: topic_model_pyspark.py [-h] [-n 10] [-i 100] [-v 5000] [-m1 0.02]
                              [-m2 0.8]

optional arguments:
  -h, --help            show this help message and exit
  -n 10, --num-topics 10
                        Number of topics in LDA
  -i 100, --iterations 100
                        Iterations in LDA
  -v 5000, --vocabsize 5000
                        <aximum vocabulary size for LDA
  -m1 0.02, --minDF 0.02
                        Minimum document frequency for LDA
  -m2 0.8, --maxDF 0.8  Maximum document frequency for LDA
```

---


