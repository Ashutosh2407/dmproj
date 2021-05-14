#PROGRAM
import pandas as pd
import numpy as np
import joblib
import re
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download("stopwords")
stopwords=stopwords.words('english')
corpusroot=r"short.csv"
doc=pd.read_csv(corpusroot,encoding="latin1")
dictionary=doc.to_dict()
document=dictionary["description"]  #CORPUS I.E LIST OF DOCS
def give_data(doc):
	for x in doc:
		return doc["description"]
#doc["description"] = doc["description"].apply(literal_eval)
doc['textual'] = doc.apply(give_data, axis = 1)
#print(doc["textual"][0])
#print(give_data(doc))


def text_proc(text):
    text = text.replace("uncredited","")
    text = re.sub('[^a-z\s]', '', text.lower())
    text = [word for word in text.split(' ') if word not in set(stopwords)]
    return ' '.join(text)

doc['textual'] = doc['textual'].apply(text_proc)
#print(doc['textual'][0])



english_stemmer = SnowballStemmer('english')
analise = CountVectorizer().build_analyzer()

def stemming(text):
    return (english_stemmer.stem(word) for word in analise(text))

count = CountVectorizer(analyzer = stemming)

count_matrix = count.fit_transform(doc['textual'])
#print(count_matrix)



tfidf_transform = TfidfTransformer()
train_tfidf = tfidf_transform.fit_transform(count_matrix)
#print(train_tfid)



def get_search(query):
    query = text_proc(query)
    query_matrix = count.transform([query])
    query_tfidf = tfidf_transform.transform(query_matrix)
    sim_score = cosine_similarity(query_tfidf, train_tfidf)
    sorted_indexes = np.argsort(sim_score).tolist()
    return doc.iloc[sorted_indexes[0][-3:]]

"""wines = get_search("ripe aroma")
print(wines)
"""
joblib.dump(count, 'count.pkl')
joblib.dump(tfidf_transform, 'tfidf.pkl')
joblib.dump(train_tfidf, 'trained_tfidf.pkl')
