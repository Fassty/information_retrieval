## Installation:

Requires:
- python >= 3.8 (walrus operator)

Then just run:
```shell
pip install -r requirements.txt
```

For `nltk` you need to download the pretrained models for `WordNetLemmatizer`:
```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
```
