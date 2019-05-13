from flask import Flask,render_template,request
import pandas as pd
import numpy as np
from sklearn.externals import joblib
import re
from nltk.tokenize import RegexpTokenizer
from program import get_search_results
from nltk.corpus import stopwords
from ast import literal_eval
from nltk.stem import WordNetLemmatizer
from naivebayes import naive_bayes
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity



# def script_output():
# 	output = execute("./program.py")
# 	return render_template("results.html",output=output)
df=pd.read_csv(r"C:\Users\Ashutosh Wagh\Desktop\winemagazine.csv",encoding="latin1")
classes=["Pinot Noir","Chardonnay","Cabernet Sauvignon","Red Blend","Bordeaux-style Red Blend"]


lemmatizer = WordNetLemmatizer()
#nltk.download("stopwords")
stop_words = stopwords.words('english')

def isClass(x):
    if x in classes:
        return 1
    else:
        return 0

def getY(y):
    arr = [0,0,0,0,0]
    arr[classes.index(y)]=1
    return arr

df["flag"] = df["variety"].apply(isClass)
new_df = df[df["flag"]==1]
new_df = new_df.sample(frac=1).reset_index(drop=True)
new_df=new_df[:5001]


x_train = new_df['description'].str.split(" ")
y_train = []
for _,y in new_df.iterrows():
	y_train.append(getY(y['variety']))

x_train = np.array(x_train)
y_train = np.array(y_train)

def stemming(text):
    return (english_stemmer.stem(w) for w in analyzer(text))



def process_test_classification(text):
    # print(type(text))   
    text = re.sub('[^a-z\s]', '', text.lower())
    text = [lemmatizer.lemmatize(w) for w in text.split() if w not in set(stop_words)]
    return ' '.join(text)

def process_text(text):
    text = re.sub('[^a-z\s]','', text.lower())
    text = [w for w in text.split() if w not in set(stop_words)]
    return ' '.join(text)

def predict(text):
	text = process_text(text)
	pred = abc.predict(text.split(' '))
	df = pd.DataFrame(data=pred,index=[0])
	return df
    

abc=naive_bayes()  
abc.initialize(x_train,y_train, list(classes))




"""@app.route("/results")
def results():
	return render_template("results.html")"""

app=Flask(__name__)
@app.route("/")
def index():
	return render_template("index.html")


@app.route("/",methods=["GET","POST"])
def getValue():
	query = request.form["query"]
	print (query)
	wines = get_search_results(query)

	return render_template("results.html",query = query,wines = wines)


@app.route("/classify")
def classify():
	return render_template("classify.html")
	
@app.route("/classify",methods=["GET","POST"])
def getVar():
	query = request.form["query"]
	df = predict(query)
	return render_template("resultclassify.html",query = query,df = df)

@app.route("/recommend",methods=["GET","POST"])
def recommend():
	query=df["title"].tolist()
	
	try:
		wine_id = request.args.get('wine_id')
		wine_id = int(wine_id)
	except Exception as e:
		return render_template("recommend.html",nums = len(query), wine_list= query)
	wines = get_search_results(df['description'].iloc[wine_id])
	return render_template('recommendresult.html', wines = wines)



if __name__=="__main__":
	app.run(host='127.0.0.1', port=5000, debug=True)
