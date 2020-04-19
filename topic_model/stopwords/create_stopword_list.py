"""
Script to generate a custom list of stopwords that extend upon existing lists.
"""
import spacy

# In this example, we first get the default spaCy stopword list
nlp = spacy.blank('en')
spacy_stopwords = nlp.Defaults.stop_words
print(f"Obtained {len(list(spacy_stopwords))} from spaCy.")

# These words are very common and do not add value to topic models
add_stopwords = set(
    ['mr', 'mrs', 'ms', 'dr', 'mr.', 'mrs.', 'ms.', 'dr.',
     'man', 'woman', 'de'
     ]
)

combined_stopwords = spacy_stopwords.union(add_stopwords)
stopword_list = sorted(list(combined_stopwords))
print(f"New stopword list contains {len(stopword_list)} words.")

# Write out stopwords to file
with open('custom_stopwords.txt', 'w') as f:
    for word in stopword_list:
        f.write(word + '\n')
