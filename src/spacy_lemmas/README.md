# Using the spaCy lemma list in Spark
A good, reliable list of English lemmas is [publicly available as part of the `spaCy-lookups` data on GitHub](https://github.com/explosion/spacy-lookups-data/tree/master/spacy_lookups_data/data). As can be seen, the data is in JSON format.

To use spaCy lemmas in another software such as Spark NLP, the JSON file is written out in the form required by Spark NLP using the script `spacy_to_spark.py`. An example is shown below. The word to the left of the delimiter `->` is the lemma form, while the words to the right are the various inflectional forms of the word.

```
task -> task tasked tasking tasks
taskmaster -> taskmaster taskmasters
tassel -> tassel tasseled tasseling tasselled tasselling tassels
taste -> taste tasted tastes tasting
taster -> taster tasters
tasty -> tasty tastier tastiest
tatami -> tatami tatamis
tatoo -> tatoo tatoos
tatter -> tatter tattered tattering tatters
tatty -> tatty tattier tattiest
tattoo -> tattoo tattooed tattooing tattoos
tattooist -> tattooist tattooists
teach -> teach taught teaches teaching
taunt -> taunt taunted taunting taunts
```

This list is used as a lookup table in Spark NLP, allowing us to perform large-scale, parallelized lemmatization.
