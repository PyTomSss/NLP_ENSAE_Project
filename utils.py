from newspaper import Article



### Function to extract text from a URL using Newspaper3k

def extract_text_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except:
        return None