import sys, argparse, string, os
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import csv
import nltk
import warnings
from utils.io import *

from nltk.translate.bleu_score import SmoothingFunction
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

def bleu(pred, gt):
    warnings.filterwarnings('ignore')
    # NLTK
    # Download Punkt tokenizer (for word_tokenize method)
    # Download stopwords (for stopword removal)
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')
    
    # English Stopwords
    stops = set(stopwords.words("english"))

    # Stemming
    stemmer = SnowballStemmer("english")

    # Remove punctuation from string
    translator = str.maketrans('', '', string.punctuation)

    pred = pred.lower()
    gt = gt.lower()

    candidate_words = nltk.tokenize.word_tokenize(pred.translate(translator))
    gt_words = nltk.tokenize.word_tokenize(gt.translate(translator))


    candidate_words = [word for word in candidate_words if word.lower() not in stops]
    gt_words = [word for word in gt_words if word.lower() not in stops]


    candidate_words = [stemmer.stem(word) for word in candidate_words]
    gt_words = [stemmer.stem(word) for word in gt_words]

    bleu_score = nltk.translate.bleu_score.sentence_bleu([gt_words], candidate_words, smoothing_function=SmoothingFunction().method0)

    return bleu_score

