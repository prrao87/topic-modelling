"""
Script to generate a custom list of stopwords that extend upon existing lists.
"""
import json
import spacy
from urllib.request import urlopen
from itertools import chain


def combine(*lists):
    "Combine an arbitrary number of lists into a single list"
    return list(chain(*lists))


def get_spacy_lemmas():
    spacy_lemma_url = "https://raw.githubusercontent.com/explosion/spacy-lookups-data/master/spacy_lookups_data/data/en_lemma_lookup.json"
    with urlopen(spacy_lemma_url) as response:
        lemmas = response.read()
    return json.loads(lemmas)


def lookup_verbs(roots, spacy_lemmas):
    """Return a full of list light verbs and all its forms"""

    def flatten(list_of_lists):
        "Return a flattened list of a list of lists"
        return [item for sublist in list_of_lists for item in sublist]

    verblist = []
    for root in roots:
        verbs = [key for key in spacy_lemmas if spacy_lemmas[key] == root]
        verbs.append(root)
        verblist.append(verbs)
    return flatten(verblist)


if __name__ == "__main__":
    # We first get the default spaCy stopword list
    nlp = spacy.blank('en')
    spacy_stopwords = nlp.Defaults.stop_words
    spacy_lemmas = get_spacy_lemmas()

    # Create custom lists depending on the class of words seen in the data
    person_titles = ['mr', 'mrs', 'ms', 'dr', 'mr.', 'mrs.', 'ms.', 'dr.']
    broken_words = ['don', 'isn', 'mustn', 'shouldn', 'couldn', 'doesn', 'didn']
    numbers = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '000']
    url_terms = ['http', 'https', 'ref', 'href', 'com']
    days_of_the_week = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday',
                        'saturday', 'sunday']
    time_periods = ['minute', 'minutes', 'hour', 'hours', 'day', 'days', 'week', 'weeks',
                    'month', 'months', 'year', 'years']
    time_related = ['yesterday', 'today', 'tomorrow', 'day', 'night', 'morning',
                    'afternoon', 'evening', 'edt', 'est', 'time', 'times']
    common_nouns = ['people', 'family', 'friend', 'father', 'mother', 'brother', 'sister',
                    'son', 'daughter', 'life', 'man', 'woman', 'way', 'thing',
                    'new', 'york', 'news']
    social_media = ['twitter', 'facebook', 'google', 'gmail', 'video', 'photo', 'image',
                    'user', 'social', 'media', 'page', 'online', 'live', 'stream', 'post',
                    'app']
    light_verb_roots = ['like', 'love', 'think', 'want', 'know', 'feel', 'look',
                        'come', 'try', 'go', 'take', 'give', 'get', 'tell', 'ask', 'talk',
                        'write', 'speak', 'say', 'comment', 'include', 'use', 'see', 'call',
                        'make', 'find']
    # Convert light verb roots to all its forms using lemma lookup
    light_verbs_full = lookup_verbs(light_verb_roots, spacy_lemmas)

    # Combine into a single lit of stopwords
    add_stopwords = set(
        combine(
            person_titles, broken_words, numbers, url_terms, days_of_the_week, 
            time_periods, time_related, common_nouns, social_media, light_verbs_full
        )
    )

    # Combine all stopwords into one list and export to text file
    combined_stopwords = spacy_stopwords.union(add_stopwords)
    stopword_list = sorted(list(combined_stopwords))
    # Write out stopwords to file
    with open('custom_stopwords.txt', 'w') as f:
        for word in stopword_list:
            f.write(word + '\n')

    print(f"Exported {len(stopword_list)} words to stopword list.")
