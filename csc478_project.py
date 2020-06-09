#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk 
import numpy as np
import pandas as pd
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
# import warnings
# warnings.filterwarnings('ignore')
import time


# <h4>Create a lemmatizer object to transform word into their base form (i.e dogs and dog becomes the same word)</h4>

# In[2]:


wordnet_lemmatizer = WordNetLemmatizer() 


# <h4>Gather in the stop words and put into a set</h4>

# In[3]:


# Gather stop words
stopwords = set(w.rstrip() for w in open('stopwords.txt'))
stopwords


# <h4>Read in the dataset used to train and test the sentiment analyzer</h4>

# In[4]:


original_df = pd.read_csv("Reddit_Data.csv") # This dataset contains ~37k arbritrary reddit comment along with their sentiment
print(original_df.shape)
original_df = original_df.dropna(axis=0) # Remove NA's
print(original_df.shape)
# df = df.head(25000)
original_df


# <h4>This function accepts a dataframe as a parameter and tokenizes every entry in the dataframe and adds it to a tokens list. It also removes tokens that are stopwords or less than 3 characters. It then adds all the unique words to a word index map where the key is the word/token and the value is the index of where it first occurred. The purpose of this word index map is to allow me to map a word to a position and compute the frequency of its occurrence. Lastly, the function returns the tokens list and the word index map</h4>

# In[5]:


# Loops through each comment and tokenize it and remove stop words. Also, create word index map to compute word frequencies
def tokenize_comments(df):
    tokens_list = [] 
    word_index_map = {}
    i=0
    for index, row in df.iterrows():
        tokens = nltk.tokenize.word_tokenize(row["clean_comment"])# Tokenize the comments
        tokens = [t for t in tokens if len(t) > 2]
        tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # Convert words into their base form
        tokens = [t for t in tokens if t not in stopwords] # Only adds words to the tokens if they are not stopwords and the length of the string is > 2
        for token in tokens: # Loop through and get add each token/word to word index map
            if token not in word_index_map:
                word_index_map[token] = i
                i += 1
        tokens_list.append(tokens)
    return tokens_list, word_index_map # return tokenize list and the word index map


# <h4>The function "normalize_tokens" accepts the tokens list and the word index map returned for the "tokenize_comments" function and normalizes each token according to its token list.</h4>

# In[6]:


# Structure and normalize each tokens list
# np.seterr(divide = 'ignore') # Ignore divide by zero warning
def normalize_tokens(tokens_list, word_index_map):
    normalized_tokens_list = []
    for token_list in tokens_list: # Loop each tokens list (comment that has been tokenized)
        x = np.zeros(len(word_index_map) + 1)
        for token in token_list: # Loop each token in the comment 
            i = word_index_map[token] # Get (first) occurence of word. Arbritray number but needed for organization
            x[i] += 1 # Increment
        x = np.divide(x,x.sum()) # Divide the frequency vector by total sum allowing for us to investigate it words total usage in the comment
        normalized_tokens_list.append(x)
    return normalized_tokens_list


# <h4>Function that appends the target variable which is the sentiment label back onto its respective tokens list </h4>

# In[7]:


def attach_labels(tokens_list, labels):
    for i in range(len(tokens_list)):
        tokens_list[i] = np.append(tokens_list[i], np.array(labels)[i])
    return tokens_list


# In[8]:


# Tokenize comments
tokens_list, word_index_map = tokenize_comments(original_df)


# In[9]:


print("Tokens list without label \n")
tokens_list


# In[10]:


print("Word index map")
word_index_map


# In[11]:


# Normalize token
normalized_tokens = normalize_tokens(tokens_list, word_index_map)


# In[12]:


print('Normalized tokens. There are twenty words in the first array and the first word "family" appeared once so it has a value of 1/20=0.05')
normalized_tokens


# <h4>Attach the sentiment labels back to its respective comment</h4>

# In[13]:


new_tokens_list = attach_labels(normalized_tokens, original_df["category"])


# <h4>Convert the list of np arrays into np mats and then into dataframe</h4>

# In[14]:


new_df = pd.DataFrame(np.mat(new_tokens_list))


# <h4>We must drop the rows that contain null/na values that came from removing stopwords and words less than 3 chars</h4>

# In[15]:


# Drop row that contains NA values
new_df = new_df.dropna(axis=0)


# <h4>Display new dataframe</h4>

# In[16]:


new_df


# <h4>Extract the target variable from the new dataframe and remove it</h4>

# In[17]:


df_target = new_df[new_df.columns[-1]] # Extract target variable 
df = new_df.drop(new_df.columns[-1], axis = 1) # Drop target column from data


# <h4>Split the dataset</h4>

# In[18]:


from sklearn.model_selection import train_test_split
df_train, df_test, df_train_target, df_test_target = train_test_split(df, df_target, test_size=0.2, random_state=33)


# <h4>Create a sentiment analyzer using logistic regression</h4>

# In[19]:


start_time = time.time()

sentiment_analyzer_lr = LogisticRegression(max_iter = 500)
sentiment_analyzer_lr.fit(df_train, df_train_target)
sentiment_analyzer_lrpreds_test = sentiment_analyzer_lr.predict(df_test)
# Run 5-fold validation
cv_scores = cross_val_score(sentiment_analyzer_lr, df_test, df_test_target, cv=5)
print("Accuracy for logistic regression classifier")
print("Overall Accuracy on X-Val: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))
print("Score on Training: ", sentiment_analyzer_lr.score(df_train, df_train_target))
print("Score on Test: ", sentiment_analyzer_lr.score(df_test, df_test_target))


print("--- %s seconds ---" % (time.time() - start_time))


# <h4>Create a sentiment analyzer using a decision tree</h4>

# In[20]:


start_time = time.time()

from sklearn import tree
sentiment_analyzer_tree = tree.DecisionTreeClassifier(criterion='gini', max_depth=3)
sentiment_analyzer_tree = sentiment_analyzer_tree.fit(df_train, df_train_target)
# Compute predictions
sentiment_analyzer_treepreds_test = sentiment_analyzer_tree.predict(df_test)
# Run 5-fold validation
cv_scores = cross_val_score(sentiment_analyzer_tree, df_train, df_train_target, cv=5)
print("Accuracy for decision tree classifier")
print("Overall Accuracy on X-Val: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))
print("Score on Training: ", sentiment_analyzer_tree.score(df_train, df_train_target))
print("Score on Test: ", sentiment_analyzer_tree.score(df_test, df_test_target))
## Possibly overfitting the training set ##

print("--- %s seconds ---" % (time.time() - start_time))


# <h4> Once the sentiment analyzer is built, let's now build the DJIA prediction system</h4>

# In[ ]:


# Gather in the dataset
data = pd.read_csv("Combined_News_DJIA.csv")
data
print("Dimension before removing NA's: " + str(data.shape))


# <h4>Drop rows that contain NA and reindex the dataframe after dropping the rows</h4>

# In[ ]:


# Drop na's and reindex dataframe
data = data.dropna()
print("Dimension after removing NA's: " + str(data.shape))
data = data.reset_index(drop=True) # Reindex the dataframe after dropping column
data


# <h4>Loop through each column and replace the unneccessary tag "b."</h4>

# In[ ]:


# Loop through each column and replace the unneccessary tag "b."
for column in data.columns[2:]: 
    data[column] = data[column].str.replace('b.', ' ', regex=True)
# data


# <h4> This function accepts a string of words (i.e. sentence/headline) tokenizes it, remove tokens that are less than 2 characters, remove tokens that are stopwords after lemmatizing the token </h4>

# In[ ]:


def tokenize_string(input_str):
    tokens = nltk.tokenize.word_tokenize(input_str) # Tokenize string
    tokens = [t for t in tokens if len(t) > 2] # Remove words less than 2 chars
    tokens = [wordnet_lemmatizer.lemmatize(t) for t in tokens] # Convert words into their base form
    tokens = [t for t in tokens if t not in stopwords] # Only adds words to the tokens if they are not stopwords and the length of the string is > 2
#     for token in tokens: # Loop through and add each token/word to word index map if not already added
#             if token not in word_index_map:
#                 word_index_map[token] = i
#                 i += 1
    return tokens


# <h4>This function creates a vector where each index represents a word in the word dictionary. At each index, the value is the frequency of that word occuring in the token's list passed in the function. Lastly, divide the frequency of each of entry in the vector by the total number of tokens (tokens in the token list) </h4>

# In[ ]:


def normalize_tokenized_string(tokens):
    x = np.zeros(len(word_index_map) + 1)
    for token in tokens: # Loop each token in the comment 
        if token in word_index_map: # We can't analyze words that we haven't used for training
            i = word_index_map[token] # Get (first) occurence of word. Arbritray number but needed for organization
            x[i] += 1
    if x.sum() != 0: # No words are in this headline have sentiment analyzer been trained on
        x = np.divide(x,x.sum())
    return x


# <h4>Function that rebuilds the original dataset that was made up of top 25 headlines for each day from June 2008 to July 2016 with it sentiment label representation. To do this, I pass a sentiment analyzer along with the dataset to make predictions of the sentiment of each headlines in every data object. Return newly created matrix. </h4>

# In[ ]:


def reconstruct(data, sentiment_analyzer):
    mat = np.zeros(shape=(data.shape[0], data.shape[1]-2)) # Create matrix to hold the data
    position = 0 # Pointer for our data
    for index, row in data.iterrows():
        predictions = [] # The current row's prediction list (i.e. That day's top 25 headlines)
        for i in range(2, len(data.columns)): # Loop through each entry in the current row
            current_comment = data[data.columns[i]][index] # Get the current comment in the row
            tokens = tokenize_string(current_comment) # Tokenize the comment
            normalized_tokens = normalize_tokenized_string(tokens) # Normalize the tokens by frequency
            prediction = sentiment_analyzer.predict([normalized_tokens]) # Predict the sentiment label for this entry 
            predictions.append(prediction[0]) # Add the label to the vector
        mat[position, :] = predictions # Add the vector (the row's sentiment label representation) into the matrix
        position += 1 # Increment row position
    return mat


# <h4>Reconstruct the original reddit dataset to be represented by its
# entries sentiment label and then convert the matrix it into a dataframe</h4>

# In[ ]:


df2 = pd.DataFrame(reconstruct(data, sentiment_analyzer_lr)) 
df2 # Display new dataframe 


# <h4>Now, let's take the reconstructed dataframe and create a classification model that predicts if the DJIA close value increases or decreases from the opening value.</h4>

# In[ ]:


# Split the dataset into train and test set
df2_train, df2_test, df2_train_target, df2_test_target = train_test_split(df2, data["Label"], test_size=0.3, random_state=33)


# <h4>Create a logistic regression model to predict the behavior of the DJIA</h4>

# In[ ]:


start_time = time.time()

# Let's create logistic regression classifer to the behavior of the DJIA closing value
from sklearn.linear_model import LogisticRegression
lrclf = LogisticRegression()
lrclf.fit(df2_train, df2_train_target)
lrpreds_test = lrclf.predict(df2_test)
# Run 10-fold validation
cv_scores = cross_val_score(lrclf, df2_test, df2_test_target, cv=5)
print("Accuracy for logistic regression classifier")
print("Overall Accuracy on X-Val: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))
print("Score on Training: ", lrclf.score(df2_train, df2_train_target))
print("Score on Test: ", lrclf.score(df2_test, df2_test_target))

print("--- %s seconds ---" % (time.time() - start_time))


# <h4>Let's see if data performs better on the top 5 headlines than the top 25 headlines</h4>

# In[ ]:


# Let's trim the dataframe with just the top 5 blog post headlines 
df3 = df2
df3.drop(df3.iloc[:, 5:], inplace = True, axis=1)
df3


# <h4>Split the dataset into train and test set</h4>

# In[ ]:


df3_train, df3_test, df3_train_target, df3_test_target = train_test_split(df3, data["Label"], test_size=0.3, random_state=33)


# <h4>Create a logisic regression model</h4>

# In[ ]:


start_time = time.time()

# Create another logistic regression clf with just the top 5 headlines
lrclf2 = LogisticRegression()
lrclf2.fit(df3_train, df3_train_target)
lrpreds_test = lrclf.predict(df2_test)
# Run 10-fold validation
cv_scores = cross_val_score(lrclf, df3_test, df3_test_target, cv=5)
print("Accuracy for logistic regression classifier (with just the top 5 headlines)")
print("Overall Accuracy on X-Val: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))
print("Score on Training: ", lrclf2.score(df3_train, df3_train_target))
print("Score on Test: ", lrclf2.score(df3_test, df3_test_target))

print("--- %s seconds ---" % (time.time() - start_time))


# <h4>Create a decision tree model</h4>

# In[ ]:


start_time = time.time()

# Create decision tree classifier and train the classifier
from sklearn import tree
treeclf = tree.DecisionTreeClassifier(criterion='gini')
treeclf = treeclf.fit(df2_train, df2_train_target)
# Compute predictions
treepreds_test = treeclf.predict(df2_test)
# Run 5-fold validation
cv_scores = cross_val_score(treeclf, df2_train, df2_train_target, cv=10)
print("Accuracy for decision tree classifier")
print("Overall Accuracy on X-Val: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))
print("Score on Training: ", treeclf.score(df2_train, df2_train_target))
print("Score on Test: ", treeclf.score(df2_test, df2_test_target))
## Possibly overfitting the training set ##

print("--- %s seconds ---" % (time.time() - start_time))


# <h4>Create a linear discriminant model</h4>

# In[ ]:


start_time = time.time()

# #  Create linear discriminant analysis classifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
ldclf = LinearDiscriminantAnalysis()
# Compute predictions
ldclf = ldclf.fit(df2_train, df2_train_target)
ldpreds_test = ldclf.predict(df2_test)
# Run 5-fold validation
cv_scores = cross_val_score(ldclf, df2_test, df2_test_target, cv=5)
print("Accuracy for linear discriminant analysis classifier")
print("Overall Accuracy on X-Val: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))
print("Score on Training: ", ldclf.score(df2_train, df2_train_target))
print("Score on Test: ", ldclf.score(df2_test, df2_test_target))

print("--- %s seconds ---" % (time.time() - start_time))


# Create an SVM model

# In[ ]:


# start_time = time.time()

# # Create SVM classifier (Linear)
# from sklearn.svm import SVC # Support vector classifier
# svm_linear = SVC(kernel='linear', C=1E10)
# svm_linear.fit(df2_train, df2_train_target)
# cv_scores = cross_val_score(svm_linear, df2_test, df2_test_target, cv=5)
# print("Accuracy for support vector machine classifier")
# print("Overall Accuracy on X-Val: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))
# print("Score on Training: ", svm_linear.score(df2_train, df2_train_target))
# print("Score on Test: ", svm_linear.score(df2_test, df2_test_target))

# print("--- %s seconds ---" % (time.time() - start_time))


# In[ ]:


# start_time = time.time()

# # Create a kernelized SVM by using RBF (radial basis function)
# svm_k = SVC(kernel='rbf', C=1E6)
# svm_k.fit(X, y)
# cv_scores = cross_val_score(svm_k, df2_test, df2_test_target, cv=5)
# print("Accuracy for support vector machine classifier")
# print("Overall Accuracy on X-Val: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))
# print("Score on Training: ", svm_k.score(df2_train, df2_train_target))
# print("Score on Test: ", svm_k.score(df2_test, df2_test_target))

# print("--- %s seconds ---" % (time.time() - start_time))


# In[ ]:




