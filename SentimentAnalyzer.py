import nltk 
import numpy as np

from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from bs4 import BeautifulSoup # Needed to parse xml

wordnet_lemmatizer = WordNetLemmatizer() # Converts words into the base forms (dogs and dog becomes the same word)

stopwords = set(w.rstrip() for w in open('stopwords.txt'))

positive_reviews = BeautifulSoup(open('electronics/positive.review').read())
positive_reviews = positive_reviews.findAll('review_text') # Look for the key 'review_text'. Needed to identify xml tag

negative_reviews = BeautifulSoup(open('electronics/negative.review').read())
negative_reviews = negative_reviews.findAll('review_text')

np.random.shuffle(positive_reviews) # Shuffle the positive reviews
positive_reviews = positive_reviews[:len(negative_reviews)] # Since we already know there are more positive reviews than negative - remove the access positive reviews


# Fuction that tokenizes and removes stop words from a string of words 
def my_tokenizer(s):
	s = s.lower() # Lower case all strings 
	tokens = nltk.tokenize.word_tokenize(s) # This is faster than using str.split()
	tokens = [t for t in tokens if len(s) > 2] # Get rid of words with length less than 2
	tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # Convert words into their base form
	tokens = [t for t in tokens if t not in stopwords] # Remove stop words

	return tokens

# create index so each word can have its own index and be lookedup fast
word_index_map = {}
current_index = 0

positive_tokenized = []
negative_tokenized = []

# Tokenizes each positive review and puts it into a nested list called positive_tokenized
for review in positive_reviews:
	tokens = my_tokenizer(review.text) # pass the review as argument but cast it to text first
	positive_tokenized.append(tokens)
	# print("here")
	# print(review.text)
	for token in tokens:
		if token not in word_index_map:
			word_index_map[token] = current_index
			current_index += 1

# Tokenizes each negative review and puts it into a nested list called positive_tokenized
for review in negative_reviews:
	tokens = my_tokenizer(review.text) # pass the review as argument but cast it to text first
	negative_tokenized.append(tokens)
	for token in tokens:
		if token not in word_index_map:
			word_index_map[token] = current_index
			current_index += 1



def tokens_to_vector(tokens, label):
	
	for t in tokens:
		i = word_index_map[t] # Get the (first) occurence location of the word/token t
		x[i] += 1 # Increment that words frequency counter
	x = x / x.sum() # Divide each word frequency by the total frequency 
	x[-1] = label # Make the last element in the list the label
	return x

N = len(positive_tokenized) + len(negative_tokenized)

data = np.zeros((N, len(word_index_map) + 1))

i=0

# Take each tokenized review and convert it into a word frequency 
# vector and place in data matrix at its respective location
for tokens in positive_tokenized:
	xy = tokens_to_vector(tokens, 1)
	data[i,:] = xy
	i+=1

for tokens in negative_tokenized:
	xy = tokens_to_vector(tokens, 0)
	data[i,:] = xy
	i+=1

np.random.shuffle(data)

X = data[:,:-1]
Y = data[:, -1]

X_train = X[:-100,]
Y_train = Y[:-100,]
X_test = X[-100:,]
Y_test = Y[-100:,]

model = LogisticRegression()
model.fit(X_train, Y_train)
print("Classification rate", model.score(X_test, Y_test))


threshold = 0.5

for word, index in word_index_map.items():
	weight = model.coef_[0][index]
	if weight > threshold or weight < -threshold:
		print(word, weight)