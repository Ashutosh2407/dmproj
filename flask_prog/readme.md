Wine Recommender System:

Development Phase I: Search

The Wine Recommender System shows top 3 wine list based on user reviews.The Wine Recommender System use Kaggle Wine Taster's review dataset.
To find the top 3 wines, the website uses inverted term frequency for finding similarities between the user query and the wine review.

Background:
In the field of information retrival,tf-idf stands for term frequency inverse document frequency.Tf idf gives the measure of how important
a word is in the document.Both tf and idf have to be calculate separately.

Program:

1)Firstly, we need to load all the necessary libraries and read the dataset which is going to be used.

2)After loading the data, we need to remove stopwords and tokenise each word.

                  def process_text(text):
                     text = re.sub('[^a-z\s]', '', text.lower())
                    text = [w for w in text.split(' ') if w not in set(stopwords)]
                    return ' '.join(text)
                    
3)Now, we have created our token. We will use tokens to find TF_IDF value. We created two functions for tf and idf which will return tf and idf scores.

            tfidf_transformer = TfidfTransformer()
            train_tfidf = tfidf_transformer.fit_transform(count_matrix)
            
4)We will use the following formula for TF_IDF score. Every every t token we will save the TF_IDF value.

![formula](https://raw.githubusercontent.com/anikx7/Game_finder_data_mining/master/Image/formula1.JPG)

5)Now, we will find TF_IDF for user query. We tokenize user uery and find TF_IDF score. We will use below formula to find TF_IDF score.
![formula](https://raw.githubusercontent.com/anikx7/Game_finder_data_mining/master/Image/formula2.JPG)

6)After finding the cosine score,we need to find the cosine similarity score between the terms and the usrer query.

7)We will then rank the the result on the basis of highest cosine similarity score between terms and user query.
