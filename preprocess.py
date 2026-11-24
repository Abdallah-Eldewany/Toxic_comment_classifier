# Same cleaning used in the notebook before training.
# The tokenizer was trained on CLEANED text, so we must clean the same way here.
import re

contractions = {
    "you're": "you are",
    "u r": "you are",
    "ur": "your",
    "don't": "do not",
    "didn't": "did not",
    "isn't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "won't": "will not",
    "i'm": "i am",
    "it's": "it is",
    "that's": "that is",
    "what's": "what is",
    "they're": "they are",
    "we're": "we are",
    "wasn't": "was not",
    "weren't": "were not",
}


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', '', text)

    # 1. expand common contractions
    for word, replacement in contractions.items():
        text = re.sub(r'\b' + word + r'\b', replacement, text)

    # 2. undo symbol disguises inside words (a$$, b!tch)
    text = re.sub(r'(?<=[a-z])@(?=[a-z])', 'a', text)
    text = re.sub(r'(?<=[a-z])\$(?=[a-z\$])', 's', text)
    text = re.sub(r'(?<=[a-z])!(?=[a-z])', 'i', text)
    text = re.sub(r'(?<=[a-z])\*(?=[a-z])', 'u', text)

    # 3. keep only letters and spaces
    text = re.sub(r'[^a-z\s]', ' ', text)

    # 4. shorten repeated letters (stuuuupid -> stuupid)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    text = re.sub(r'\s+', ' ', text).strip()
    return text
