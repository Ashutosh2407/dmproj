#PROGRAM 
import pandas as pd
import numpy as np
from sklearn.externals import joblib
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


def process_text(text):
    text = text.replace("uncredited","")
    text = re.sub('[^a-z\s]', '', text.lower())
    text = [w for w in text.split(' ') if w not in set(stopwords)]
    return ' '.join(text)

doc['textual'] = doc['textual'].apply(process_text)
#print(doc['textual'][0])



english_stemmer = SnowballStemmer('english')
analyzer = CountVectorizer().build_analyzer()

def stemming(text):
    return (english_stemmer.stem(w) for w in analyzer(text))

count = CountVectorizer(analyzer = stemming)

count_matrix = count.fit_transform(doc['textual'])
#print(count_matrix)



tfidf_transformer = TfidfTransformer()
train_tfidf = tfidf_transformer.fit_transform(count_matrix)
#print(train_tfid)



def get_search_results(query):
    query = process_text(query)
    query_matrix = count.transform([query])
    query_tfidf = tfidf_transformer.transform(query_matrix)
    sim_score = cosine_similarity(query_tfidf, train_tfidf)
    sorted_indexes = np.argsort(sim_score).tolist()
    return doc.iloc[sorted_indexes[0][-3:]]
   
"""wines = get_search_results("ripe aroma")
print(wines)
"""
joblib.dump(count, 'count.pkl')
joblib.dump(tfidf_transformer, 'tfidf.pkl')
joblib.dump(train_tfidf, 'trained_tfidf.pkl')