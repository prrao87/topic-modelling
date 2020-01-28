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
        self.hashtag_pattern = re.compile(r'#[a-zA-Z0-9_]{1,50}')

    def clean_patterns(self, text):
        """Cleanup email addresses, handles and hashtags
           This is to avoid polluting the topic model with unnecessary online artefacts.
        """
        text = self.email_and_handle_pattern.sub('', text)
        text = self.hashtag_pattern.sub('', text)
        return text

    def remove_artefacts(self, text):
        """Text cleaning function inspired by the cleanup utility function in fastai.text.transform:
        https://github.com/fastai/fastai/blob/2c5eb1e219f55773330d6aa1649f3841e39603ff/fastai/text/transform.py#L58
        """
        re1 = re.compile(r'  +')
        text = text.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
              'nbsp;', ' ').replace('#36;', '$').replace('\\n', " ").replace('\n', " ").replace(
              'quot;', "'").replace('<br />', "\n").replace('\\"', '"').replace('\\xa0', ' ').replace(
              ' @.@ ', '.').replace('\xa0', ' ').replace(' @-@ ', '-').replace('\\', ' \\ ').replace(
              '“', '').replace('”', '').replace('’', '').replace('•', '').replace('—', '').replace(".", "")
        return re1.sub(' ', html.unescape(text))

    def lemmatize(self, text, add_removed_words=None):
        "Lemmatize text using a spaCy pipeline"
        stopwords = self.sentencizer.Defaults.stop_words
        if add_removed_words:
            stopwords = stopwords.union(add_removed_words)
        # Perform cleanup tasks
        text = self.clean_patterns(text)
        text = self.remove_artefacts(text)
        # text = self.cleanup(text)
        doc = self.sentencizer(text)
        lemmas = [str(tok.lemma_).lower() for tok in doc
                  if tok.is_alpha and tok.text.lower() not in stopwords]
        return lemmas


# Test cleanup functiosn on an example
if __name__ == "__main__":
    text = "This is an example@abc.com email address. I'm in the U.K. I've taken 10,000 items. <br />  \
            I also have a phone number: 301-376-5784."
    proc = TextPreprocessor()
    text = proc.clean_patterns(text)
    print(text)
    text = proc.remove_artefacts(text)
    print(text)
    text = proc.lemmatize(text)
    print(text)
