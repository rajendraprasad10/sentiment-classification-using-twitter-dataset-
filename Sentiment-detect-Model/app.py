# all required packages.
from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib
from flask_bootstrap import Bootstrap


app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	df = pd.read_csv('Review.csv')
	#df.rename(columns={"Wow... Loved this place.": "Review", "1": "sentiment"}, inplace=True)
	# Features and Labels
	X = df['Review']
	y = df['sentiment']
	
	# Extract Feature With CountVectorizer
	cv = CountVectorizer()
	X = cv.fit_transform(X) # Fit the Data
	from sklearn.model_selection import train_test_split
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=0)
	#Naive Bayes Classifier
	from sklearn.naive_bayes import MultinomialNB
	clf = MultinomialNB()
	clf.fit(X_train,y_train)
	clf.score(X_test,y_test)
	#Alternative Usage of Saved Model
	joblib.dump(clf, 'NB_sentimet_model.pkl')
	NB_spam_model = open('NB_sentimet_model.pkl','rb')
	clf = joblib.load(NB_spam_model)

	if request.method == 'POST':
		review = request.form['Review']
		data = [review]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('home.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)