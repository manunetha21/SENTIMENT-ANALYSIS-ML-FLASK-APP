from flask import Flask, request, render_template , jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from string import punctuation
import re
from nltk.corpus import stopwords

nltk.download('stopwords')

set(stopwords.words('english'))

app = Flask(__name__)

@app.route('/')
def my_form():
    return render_template('form.html')

@app.route('/predict', methods=['GET'])
def my_Work_Model():
    
    text1=request.args.get('query')
    

    stop_words1= stopwords.words('english')

    text_final1 = ''.join(c for c in text1 if not c.isdigit())

    processed_doc2 = ' '.join([word for word in text_final1.split() if word not in stop_words1])

    sa = SentimentIntensityAnalyzer()
    dd = sa.polarity_scores(text=processed_doc2)
    compound = round((1 + dd['compound'])/2, 2)

    final=compound
    text1=text_final1
    text2=dd['pos'],
    text5=dd['neg'],
    text4=compound,
    text3=dd['neu']

    return jsonify({
        "pos": text2,
        "neg": text5,
        "compound" : text4,
        "neutral" : text3,
        "text" : text1
    }) 

@app.route('/', methods=['POST'])
def my_form_post():
    stop_words = stopwords.words('english')
    
    #convert to lowercase
    text1 = request.form['text1'].lower()
    
    text_final = ''.join(c for c in text1 if not c.isdigit())
    
    #remove punctuations
    #text3 = ''.join(c for c in text2 if c not in punctuation)
        
    #remove stopwords    
    processed_doc1 = ' '.join([word for word in text_final.split() if word not in stop_words])

    sa = SentimentIntensityAnalyzer()
    dd = sa.polarity_scores(text=processed_doc1)
    compound = round((1 + dd['compound'])/2, 2)
    

    return render_template('form.html', final=compound, text1=text_final,text2=dd['pos'],text5=dd['neg'],text4=compound,text3=dd['neu'])

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5002, debug=False)
