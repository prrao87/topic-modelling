import sys
from math import ceil
from tqdm import tqdm
from typing import List, Dict, Any
from pyspark.sql import SparkSession
import pyspark.sql.functions as f
from pyspark.sql.types import (
    StructType, StructField, StringType, DateType
)
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    CountVectorizer, RegexTokenizer,
    IDF, StopWordsRemover
)
from pyspark.ml.clustering import LDA
#Spark NLP
from sparknlp.annotator import *
from sparknlp.base import *
# plotting
from matplotlib import pyplot as plt
from wordcloud import WordCloud

# Schema for input Data
schema = StructType([
    StructField("date", DateType(), True),
    StructField("headline", StringType(), True),
    StructField("content", StringType(), True)]
)
# Input file and Stopwords
inputfile = "../data/nytimes.tsv"
stopwordfile = "stopwords/custom_stopwords.txt"


def read_data(inputfile: str):
    """Read in a tab-separated file with date, headline and news content
       to a Spark DataFrame
    """
    df = (spark.read.option("header", "false")
          .csv(inputfile, sep='\t', schema=schema)
          .toDF("date", "headline", "content")
          .drop("headline"))
    return df


def set_regextokenizer(inputCol: str, outputCol: str):
    "Use Spark to perform custom Regex tokenization"
    tokenizer = RegexTokenizer(
        inputCol=inputCol,
        outputCol=outputCol,
        pattern=r"[A-Za-z0-9\-]{3,50}",  # only alphanumeric hyphenated text with 3 or more chars
        gaps=False
    )
    return tokenizer


def set_stopword_remover(stopwordfile: str, inputCol: str, outputCol: str):
    "use Spark to drop stopwords from a sequence of tokens"
    stopwords = sc.textFile(stopwordfile).collect()
    remover = (StopWordsRemover(
        inputCol=inputCol, outputCol=outputCol).setStopWords(stopwords))
    return remover


def set_document_assembler(inputCol: str):
    "Spark NLP document assembler"
    return DocumentAssembler().setInputCol(inputCol)


def set_tokenizer(inputCol: str, outputCol: str):
    "Tokenize text for input to the lemmatizer"
    tokenizer = (Tokenizer()
        .setInputCols([inputCol])
        .setOutputCol(outputCol)
    )
    return tokenizer


def set_lemmatizer(inputCol: str, outputCol: str):
    "Retrieve root lemmas out of the input tokens"
    # Use default SparkNLP English pretrained lemmatizer for now
    lemmatizer = (LemmatizerModel.pretrained(name="lemma_antbnc", lang="en")
        .setInputCols([inputCol])
        .setOutputCol(outputCol)
    )
    return lemmatizer


def set_finisher(finishedCol: str):
    "Finisher transform for Spark NLP pipeline"
    finisher = (Finisher()
        .setInputCols([finishedCol])
        .setIncludeMetadata(False)
    )
    return finisher


def set_countvectorizer(inputCol: str, outputCol: str, params: Dict[str, Any]):
    countvectorizer = CountVectorizer(
        inputCol=inputCol,
        outputCol=outputCol,
        vocabSize=params['vocabsize'],
        minDF=params['minDF'],
        maxDF=params['maxDF'],
        minTF=1.0
    )
    return countvectorizer


def set_idf(inputCol: str, outputCol: str):
    return IDF(inputCol="features", outputCol="idf")


def set_lda_model(params: Dict[str, Any]):
    lda = LDA(
        k=params['num_topics'],
        maxIter=params['iterations'],
        optimizer="online",
        seed=1,
        learningOffset=100.0,  # If high, early iterations are downweighted during training
        learningDecay=0.51,    # Set between [0.5, 1) to guarantee asymptotic convergence
    )
    return lda


def run_spark_preproc_pipeline(df):
    """Create a Spark preprocessing that transforms the input DataFrame to produce a
       final output column storing each document as a list of tokens with stopwords removed.
    """
    preprocPipeline = Pipeline(stages=[
        set_regextokenizer("content", "words"),
        set_stopword_remover(stopwordfile, "words", "words_filtered")
    ])
    wordsDF = preprocPipeline.fit(df).transform(df)
    # Concatenate list of words into a single word for downstream Spark-NLP pipeline
    preprocDF = wordsDF.withColumn("words_joined", f.concat_ws(" ", "words_filtered"))
    return preprocDF


def run_sparknlp_pipeline(df):
    """Create a SparkNLP pipeline that transforms the input DataFrame to procude a final output
       column storing each document as a sequence of lemmas (root words).
    """
    nlpPipeline = Pipeline(stages=[
        set_document_assembler("words_joined"),
        set_tokenizer("document", "token"),
        set_lemmatizer("token", "lemma"),
        set_finisher("lemma")
    ])
    nlpPipelineDF = (nlpPipeline.fit(df)
        .transform(df)
        .withColumnRenamed('finished_lemma', 'allTokens')
    )
    return nlpPipelineDF


def run_ml_pipeline(nlpPipelineDF, params: Dict[str, Any]):
    """Create a Spark ML pipeline and transform the input NLP-transformed DataFrame 
       to produce a trained LDA topic model for the given data.
    """
    mlPipeline = Pipeline(
        stages=[
            set_countvectorizer("allTokens", "features", params),
            set_idf("features", "idf"),
            set_lda_model(params)
        ]
    )
    mlModel = mlPipeline.fit(nlpPipelineDF)
    ldaModel = mlModel.stages[2]
    # Calculate upper bound on model perplexity
    mlPipelineDF = mlModel.transform(nlpPipelineDF)
    ldaPerplexity = ldaModel.logPerplexity(mlPipelineDF)
    return mlModel, ldaPerplexity


def describe_topics(mlModel) -> List[Dict[str, float]]:
    """Obtain topic words and weights from the LDA model.
       Returns: topics -> List[Dict[str, float]]
       A list of mappings between the top 15 topic words (str) and their weights
       (float) for each topic. The length of the list equals the number of topics.
    """
    # Store vocab from CountVectorizer
    vocab = mlModel.stages[0].vocabulary
    # Store LDA model part of pipeline
    ldaModel = mlModel.stages[2]

    # Take top 15 words in each topic
    topics = ldaModel.describeTopics(15)
    topics_rdd = topics.rdd

    topic_words = topics_rdd \
        .map(lambda row: row['termIndices']) \
        .map(lambda idx_list: [vocab[idx] for idx in idx_list]) \
        .collect()

    topic_weights = topics_rdd \
        .map(lambda row: row['termWeights']) \
        .collect()

    # Store topic words and weights as a list of dicts
    topics = [dict(zip(words, weights))
              for words, weights in zip(topic_words, topic_weights)]
    return topics


def plot_wordclouds(topics: List[Dict[str, float]], colormap: str="cividis") -> None:
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
        ax = fig.add_subplot((num_topics / 5) + 1, 5, idx + 1)  # Always have 5 columns in subplot
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
    fig.savefig("pyspark-topics.png", bbox_extra_artists=[st], bbox_inches='tight')


def main(params: Dict[str, Any]) -> List[Dict[str, float]]:
    df = read_data(inputfile)
    preprocDF = run_spark_preproc_pipeline(df)
    # Persist NLP DataFrame for performance
    nlpPipelineDF = run_sparknlp_pipeline(preprocDF).persist()
    mlModel, ldaPerplexity = run_ml_pipeline(nlpPipelineDF, params)
    topics = describe_topics(mlModel)
    return topics


if __name__ == "__main__":

    arg_names = ["num_topics", "iterations", "vocabsize", "minDF", "maxDF"]

    if len(sys.argv[1:]) != len(arg_names):
        raise Exception(
            "Please specify values for five LDA params: {}".format(
                ', '.join(arg_names))
        )
    parse_args = [
        int(sys.argv[1]),
        int(sys.argv[2]),
        int(sys.argv[3]),
        float(sys.argv[4]),
        float(sys.argv[5]),
    ]
    # Store LDA params as a dict
    params = dict(zip(arg_names, parse_args))

    spark = (SparkSession.builder
        .appName("Spark Topic Model")
        .master("local[*]")
        .config("spark.driver.memory", "1G")
        .config("spark.executor.memory", "1G")
        .config("spark.sql.shuffle.partitions", 64)
        .config("spark.shuffle.io.maxRetries", 20)
        .config("spark.shuffle.io.retryWait", "20s")
        .config("spark.buffer.pageSize", "2m")
        .getOrCreate()
    )
    sc = spark.sparkContext

    topics = main(params)
    plot_wordclouds(topics)

    # Close spark
    spark.stop()
