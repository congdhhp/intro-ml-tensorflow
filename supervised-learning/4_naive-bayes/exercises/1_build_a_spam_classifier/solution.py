# TODO: Step 1.1: Understanding our dataset
"""
Instructions:
- Import the dataset into a pandas dataframe using the read_table method. The file has already been
  downloaded, and you can access it using the filepath 'smsspamcollection/SMSSpamCollection'. Because this is a tab
  separated dataset we will be using '\t' as the value for the 'sep' argument which specifies this format.
- Also, rename the column names by specifying a list ['label', 'sms_message']
  to the 'names' argument of read_table().
- Print the first five values of the dataframe with the new column names.
"""
import pandas as pd

# Dataset available using filepath 'smsspamcollection/SMSSpamCollection'
df = pd.read_table('smsspamcollection/SMSSpamCollection',
                   sep='\t',
                   header=None,
                   names=['label', 'sms_message'])

# Output printing out first 5 rows
print(df.head())

# TODO: Step 1.2: Data Preprocessing
"""
Instructions:
- Convert the values in the 'label' column to numerical values using map method as follows: {'ham':0, 'spam':1} 
  This maps the 'ham' value to 0 and the 'spam' value to 1.
- Also, to get an idea of the size of the dataset we are dealing with, print out number of rows 
  and columns using 'shape'.
"""
df['label'] = df.label.map({'ham': 0, 'spam': 1})
print(df.shape)
print(df.head())  # returns (rows, columns)

# TODO: Step 2.1: Bag of Words
"""
We'll convert a collection of documents to a matrix, with each document being a row 
and each word (token) being the column, and the corresponding (row, column) values being the frequency of occurrence 
of each word or token in that document.

To handle this, we will be using sklearn's count vectorizer method which does the following:
- It tokenizes the string (separates the string into individual words) and gives an integer ID to each token.
- It counts the occurrence of each of those tokens.
"""

# TODO: Step 3.1: Training and testing sets
"""
Instructions: Split the dataset into a training and testing set by using the train_test_split method in sklearn. 
Split the data using the following variables:
- X_train is our training data for the 'sms_message' column.
- y_train is our training data for the 'label' column
- X_test is our testing data for the 'sms_message' column.
- y_test is our testing data for the 'label' column Print out the number of rows we have in each our training 
  and testing data.
"""
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df['sms_message'],
                                                    df['label'],
                                                    random_state=1)

print('Number of rows in the total set: {}'.format(df.shape[0]))
print('Number of rows in the training set: {}'.format(X_train.shape[0]))
print('Number of rows in the test set: {}'.format(X_test.shape[0]))

# TODO: Step 3.2: Applying Bag of Words processing to our dataset.
from sklearn.feature_extraction.text import CountVectorizer

# Instantiate the CountVectorizer method
count_vector = CountVectorizer()

# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)

# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
testing_data = count_vector.transform(X_test)

# TODO: Step 5: Naive Bayes implementation using scikit-learn
from sklearn.naive_bayes import MultinomialNB
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)

predictions = naive_bayes.predict(testing_data)

# TODO: Step 6: Evaluating our model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
print('Precision score: ', format(precision_score(y_test, predictions)))
print('Recall score: ', format(recall_score(y_test, predictions)))
print('F1 score: ', format(f1_score(y_test, predictions)))
