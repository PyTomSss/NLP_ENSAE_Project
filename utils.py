import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

from newspaper import Article
import re
import nltk
from nltk.tokenize import TreebankWordTokenizer, TweetTokenizer
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = list(set(stopwords.words('french')))
from string import punctuation
punctuation = list(punctuation)



### Function to extract text from a URL using Newspaper3k

def extract_text_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except:
        return None
    


### Functions to preprocess and clean texts

stopwords = stopwords + punctuation + ['“','’', '“', '”', '‘','...']

def lowerizer(article):
  """
  Lowerize a given text
  """
  return article.lower()

def remove_url(article):
    """
    Remove URL tags 
    """
    article = re.sub(r'https?:\/\/.\S+', "", article)
    return article

def remove_hashtags(article):
    """
    Remove hashtags from a given text
    """
    article = re.sub("#"," ",article)
    return article

def remove_a(article):
    """
    Remove twitter account references @ rom a given text
    """
    article = re.sub("@"," ",article)
    return article

def remove_brackets(article):
    """
    Remove square brackets from a given text 
    """
    article = re.sub('\[[^]]*\]', '', article)
    return article

def remove_stop_punct(article):
    """
    Remove punctuation and stopwords from a given text
    """
    final_article = []
    for i in article.split():
        if i not in stopwords:
            final_article.append(i.strip())
    return " ".join(final_article)

def preprocessing(article):
    """
    Computes the above-define steps to clean a given text
    """
    article = lowerizer(article)
    article = remove_url(article)
    article = remove_hashtags(article)
    article = remove_a(article)
    article = remove_brackets(article)
    article = remove_stop_punct(article)
    return article



## Function to prepare the matrix for Fleiss' Kappa

def prepare_matrix_for_fleiss(df_binary):
    
    counts_1 = df_binary * 8
    counts_0 = 8 - counts_1
    return np.vstack([counts_0, counts_1]).T



## Function to compute the most common words in our corpus

def most_common(corpus, nb_words):
  """
  Returns the chosen number of most common words in our corpus
  with their occurences number
  """
  
  articles = corpus.str.split()
  #print(articles)
  words = np.array([word for article in articles for word in article if word.lower() not in stopwords])
  counter = Counter(words)
  
  d = pd.DataFrame(counter, index=['occurrences']).transpose().reset_index()
  d.columns=['word', 'occurences']
  d = d.sort_values('occurences', ascending=False)
  
  return d[:nb_words]



## Function to compute the most common n-grams in our corpus

def get_ngrams(corpus, nb_grams, nb_words):
    """
    Computes and return the chosen most n_grams of a corpus 
    """
    count = CountVectorizer(ngram_range=(nb_grams, nb_grams)).fit(corpus)
    ensemble = count.transform(corpus).sum(axis=0)
    words_freq = [(word, ensemble[0, idx]) for word, idx in count.vocabulary_.items()]
    words_freq = sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:nb_words]



## Function to compute the most common n-grams in our corpus

def plot_unique_ngrams(ngrams_dict, num_to_show=20):
    """
    Plot side-by-side line plots of the most frequent n-grams (1, 2, 3),
    filtering out those that appear as subparts of longer n-grams.
    """

    df_dict = {
        n: pd.DataFrame(data, columns=['ngram', 'count'])
        for n, data in ngrams_dict.items()
    }

    higher_ngrams = {
        1: df_dict.get(2, pd.DataFrame())['ngram'].tolist() + df_dict.get(3, pd.DataFrame())['ngram'].tolist(),
        2: df_dict.get(3, pd.DataFrame())['ngram'].tolist(),
        3: []
    }

    for n in df_dict:
        exclusion = set(higher_ngrams[n])
        df_dict[n] = df_dict[n][~df_dict[n]['ngram'].apply(lambda x: any(x in h for h in exclusion))]

    fig, axes = plt.subplots(1, len(df_dict), figsize=(18, 8), sharey=True)

    for idx, (n, df) in enumerate(sorted(df_dict.items())):
        df = df.sort_values('count').tail(num_to_show)

        axes[idx].plot(df['count'], df['ngram'], lw=4)
        axes[idx].set_title(f"Unique {n}-grams", fontsize=18)
        axes[idx].set_xlabel("Occurrence count", fontsize=14)
        axes[idx].tick_params(labelsize=12)

        if idx == 0:
            axes[idx].set_ylabel("n-grams", fontsize=14)

    plt.tight_layout()
    plt.show()



## Function to analyze a text with the Gate API

def analyze_with_gate(text):
    try:
        response = requests.post(
            SERVICE_URL,
            headers=headers,
            data=text.encode('utf-8'),  # texte brut
            timeout=15
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        #print("❌ Error:", e)
    
    return None