"""
Text preprocessing utilities for topic models
"""
import spacy
import re
import html


class TextPreprocessor:
    def __init__(self):
        sentencizer = spacy.blank('en')
        sentencizer.add_pipe(sentencizer.create_pipe('sentencizer'))
        self.sentencizer = sentencizer
        # Compile Regex patterns
        self.email_and_handle_pattern = re.compile(r'(\S*@\S*)')
        self.domain_pattern = re.compile(r'(\S*\.\S*)')
        self.hashtag_pattern = re.compile(r'#[a-zA-Z0-9_]{1,50}')

    def clean_patterns(self, text):
        """Cleanup email addresses, handles and hashtags
           This is to avoid polluting the topic model with unnecessary online artefacts.
        """
        text = self.email_and_handle_pattern.sub('', text)
        text = self.domain_pattern.sub('', text)
        text = self.hashtag_pattern.sub('', text)
        return text

    def cleanup(self, text):
        """Text cleaning function inspired by the cleanup utility function in fastai.text.transform:
        https://github.com/fastai/fastai/blob/2c5eb1e219f55773330d6aa1649f3841e39603ff/fastai/text/transform.py#L58
        """
        re1 = re.compile(r'  +')
        text = text.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
              'nbsp;', ' ').replace('#36;', '$').replace('\\n', " ").replace('\n', " ").replace(
              'quot;', "'").replace('<br />', "\n").replace('\\"', '"').replace('\\xa0', ' ').replace(
              ' @.@ ', '.').replace('\xa0', ' ').replace(' @-@ ', '-').replace('\\', ' \\ ').replace(
              '“', '').replace('”', '').replace('’', '').replace('•', '').replace('—', '')
        return re1.sub(' ', html.unescape(text))

    def tokenize(self, text):
        "Tokenize input string using a spaCy pipeline"
        doc = self.sentencizer(text)
        tokenized_text = ' '.join(token.text for token in doc)
        return tokenized_text

    def lemmatize(self, text, add_removed_words):
        "Lemmatize text using a spaCy pipeline"
        stopwords = self.sentencizer.Defaults.stop_words
        combined_stopwords = stopwords.union(add_removed_words)
        text = self.clean_patterns(text)
        text = self.cleanup(text)
        doc = self.sentencizer(text)
        lemmas = [str(tok.lemma_).lower() for tok in doc
                  if tok.text.lower() not in combined_stopwords]
        return lemmas