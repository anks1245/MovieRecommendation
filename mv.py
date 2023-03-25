import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from sklearn.metrics.pairwise import cosine_similarity

from flask import Flask,jsonify,render_template
import json

app = Flask(__name__)

movies_ds = pd.read_csv('imdb_top_1000.csv')
movies_ds.isnull().sum()
movies_ds.dropna()
# movies_ds['Series_Title']

cv = CountVectorizer(max_features=5000,stop_words="english")
cv.fit_transform(movies_ds['Overview']).toarray().shape
vectors = cv.fit_transform(movies_ds['Overview']).toarray()

def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

movies_ds['Overview'] = movies_ds['Overview'].apply(stem)

similarity = cosine_similarity(vectors)

# print(movies_ds)
@app.route('/')
def landing():
    name = "json.dumps(movies_ds) "
    listOfMovies = []
    for index,row in movies_ds.iterrows():
        # print(index)
        if index == 100:
            break
        listOfMovies.append({
            "movie_name":row.Series_Title,
            "movie_poster":row.Poster_Link,
            "released_year":row.Series_Title,
            "run_time":row.Runtime,
            # "certificate":movies_ds.iloc[i[0]].Certificate,
            "genre":row.Genre,
            "IMDB_rating":row.IMDB_Rating,
            "overview":row.Overview,
            # "meta_score":movies_ds.iloc[i[0]].Series_Title,
            # "Gross":movies_ds.iloc[i[0]].Poster_Link,
        })
        # print(listOfMovies[index])
    return render_template("./index.html", movies=movies_ds[:100])

@app.route('/movie/<name>')
def recommendation(name):
    print(name)
    try:
            movie_index = movies_ds[movies_ds['Series_Title']==name].index[0]
            distances = similarity[movie_index]
            movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]
            listOfMovies = []
            # # print(movies_list)
            for i in movies_list:
                # print(movies_ds.iloc[i[0]].Series_Title)
                listOfMovies.append(movies_ds.iloc[i[0]])
                print(movies_ds.iloc[i[0]])
                print("---------------------------------")
            print(listOfMovies)
            return render_template("./movie.html", listOfMovies=listOfMovies,movie_name = name)
            # return json.dumps(listOfMovies)
    except Exception as e: 
        print("an error occurred")  
        print(e.with_traceback())
        return json.dumps({"error":"Moive is not present in our dataset"})

    
# mov = input('Enter a movie name :')
# recommendation(mov)

if __name__ == '__main__':
    app.run(debug=True)





