from __future__ import unicode_literals
from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap 

from spacy_summarization import text_summarizer
from gensim.summarization import summarize
from nltk_summarization import nltk_summarizer
import time
import spacy

# NLP Packages
from textblob import TextBlob,Word 
import random 

nlp = spacy.load('en_core_web_sm')
app = Flask(__name__)
Bootstrap(app)

# Web Scraping Pkg
from bs4 import BeautifulSoup
# from urllib.request import urlopen
import urlopen

# Sumy Pkg
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# Sumy 
def sumy_summary(docx):
	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	lex_summarizer = LexRankSummarizer()
	summary = lex_summarizer(parser.document,3)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result

# Reading Time
def readingTime(mytext):
	total_words = len([ token.text for token in nlp(mytext)])
	estimatedTime = total_words/200.0
	return estimatedTime

# Fetch Text From Url
def get_text(url):
	page = urlopen(url)
	soup = BeautifulSoup(page)
	fetched_text = ' '.join(map(lambda p:p.text,soup.find_all('p')))
	return fetched_text


@app.route('/')
def Homepage():
	return render_template('Homepage.html')


@app.route('/index')
def index():
	return render_template('index.html')


@app.route('/analyze',methods=['GET','POST'])
def analyze():
	start = time.time()
	if request.method == 'POST':
		rawtext = request.form['rawtext']
		final_reading_time = readingTime(rawtext)
		final_summary = text_summarizer(rawtext)
		summary_reading_time = readingTime(final_summary)
		end = time.time()
		final_time = end-start
	return render_template('index.html',ctext=rawtext,final_summary=final_summary,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time)


@app.route('/compare_summary')
def compare_summary():
	return render_template('compare_summary.html')

@app.route('/comparer',methods=['GET','POST'])
def comparer():
	start = time.time()
	if request.method == 'POST':
		rawtext = request.form['rawtext']
		final_reading_time = readingTime(rawtext)
		final_summary_spacy = text_summarizer(rawtext)
		summary_reading_time = readingTime(final_summary_spacy)
		# Gensim Summarizer
		final_summary_gensim = summarize(rawtext)
		summary_reading_time_gensim = readingTime(final_summary_gensim)
		# NLTK
		final_summary_nltk = nltk_summarizer(rawtext)
		summary_reading_time_nltk = readingTime(final_summary_nltk)
		# Sumy
		final_summary_sumy = sumy_summary(rawtext)
		summary_reading_time_sumy = readingTime(final_summary_sumy) 

		end = time.time()
		final_time = end-start
	return render_template('compare_summary.html',ctext=rawtext,final_summary_spacy=final_summary_spacy,final_summary_gensim=final_summary_gensim,final_summary_nltk=final_summary_nltk,final_time=final_time,final_reading_time=final_reading_time,summary_reading_time=summary_reading_time,summary_reading_time_gensim=summary_reading_time_gensim,final_summary_sumy=final_summary_sumy,summary_reading_time_sumy=summary_reading_time_sumy,summary_reading_time_nltk=summary_reading_time_nltk)


@app.route('/analyzer.html')
def analyzer():
	return render_template('analyzer.html')

@app.route('/analyse',methods=['POST'])
def analyse():
	start = time.time()
	if request.method == 'POST':
		rawtext = request.form['rawtext']
		#NLP Stuff
		blob = TextBlob(rawtext)
		received_text2 = blob
		blob_sentiment,blob_subjectivity = blob.sentiment.polarity ,blob.sentiment.subjectivity
		number_of_tokens = len(list(blob.words))
		# Extracting Main Points
		nouns = list()
		for word, tag in blob.tags:
		    if tag == 'NN':
		        nouns.append(word.lemmatize())
		        len_of_words = len(nouns)
		        rand_words = random.sample(nouns,len(nouns))
		        final_word = list()
		        for item in rand_words:
		        	word = Word(item).pluralize()
		        	final_word.append(word)
		        	summary = final_word
		        	end = time.time()
		        	final_time = end-start


	return render_template('analyzer.html',received_text = received_text2,number_of_tokens=number_of_tokens,blob_sentiment=blob_sentiment,blob_subjectivity=blob_subjectivity,summary=summary,final_time=final_time)


@app.route('/about')
def about():
	return render_template('index.html')

if __name__ == '__main__':
	app.run(debug=True)
