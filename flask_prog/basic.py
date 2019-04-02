from flask import Flask,render_template,request
import pandas as pd
import numpy as np
from sklearn.externals import joblib
import re
from nltk.tokenize import RegexpTokenizer
from program import get_search_results
from nltk.corpus import stopwords
from ast import literal_eval
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

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
		
# def script_output():
# 	output = execute("./program.py")
# 	return render_template("results.html",output=output)

@app.route("/results")
def results():
	return render_template("results.html")


if __name__=="__main__":
	app.run()