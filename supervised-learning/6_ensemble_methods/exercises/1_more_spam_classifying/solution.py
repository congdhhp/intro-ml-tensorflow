# Import our libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Read in our dataset
df = pd.read_table('smsspamcollection/SMSSpamCollection',
                   sep='\t',
                   header=None,
                   names=['label', 'sms_message'])

# Fix our response value
df['label'] = df.label.map({'ham': 0, 'spam': 1})

# Split our dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(df['sms_message'],
                                                    df['label'],
                                                    random_state=1)

# Instantiate the CountVectorizer method
count_vector = CountVectorizer()

# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)

# Transform testing data and return the matrix. Note we are not fitting the testing data into the CountVectorizer()
testing_data = count_vector.transform(X_test)

# Instantiate our model
naive_bayes = MultinomialNB()

# Fit our model to the training data
naive_bayes.fit(training_data, y_train)

# Predict on the test data
predictions = naive_bayes.predict(testing_data)

# Score our model
print('Accuracy score: ', format(accuracy_score(y_test, predictions)))
print('Precision score: ', format(precision_score(y_test, predictions)))
print('Recall score: ', format(recall_score(y_test, predictions)))
print('F1 score: ', format(f1_score(y_test, predictions)))

# TODO: Step 1: First use the documentation to import all three of the models.
# Import the Bagging, RandomForest, and AdaBoost Classifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier

# TODO: Step 2: Now that you have imported each of the classifiers, instantiate each with the hyperparameters specified
#  in each comment. In the upcoming lessons, you will see how we can automate the process
#  to finding the best hyperparameters. For now, let's get comfortable with the process and our new algorithms.
# Instantiate a BaggingClassifier with:
# 200 weak learners (n_estimators) and everything else as default values
bag_mod = BaggingClassifier(n_estimators=200)

# Instantiate a RandomForestClassifier with:
# 200 weak learners (n_estimators) and everything else as default values
rf_mod = RandomForestClassifier(n_estimators=200)

# Instantiate an AdaBoostClassifier with:
# With 300 weak learners (n_estimators) and a learning_rate of 0.2
ada_mod = AdaBoostClassifier(n_estimators=300, learning_rate=0.2)

# TODO: Step 3: Now that you have instantiated each of your models, fit them using the training_data and y_train.
#  This may take a bit of time, you are fitting 700 weak learners after all!

# Fit your BaggingClassifier to the training data
bag_mod.fit(training_data, y_train)

# Fit your RandomForestClassifier to the training data
rf_mod.fit(training_data, y_train)

# Fit your AdaBoostClassifier to the training data
ada_mod.fit(training_data, y_train)

# TODO: Step 4: Now that you have fit each of your models, you will use each to predict on the testing_data.
# Predict using BaggingClassifier on the test data
bag_preds = bag_mod.predict(testing_data)

# Predict using RandomForestClassifier on the test data
rf_preds = rf_mod.predict(testing_data)

# Predict using AdaBoostClassifier on the test data
ada_preds = ada_mod.predict(testing_data)


# TODO: Step 5: Now that you have made your predictions, compare your predictions to the actual values using
#  the function below for each of your models - this will give you the score for how well each of your models
#  is performing. It might also be useful to show the naive bayes model again here.
def print_metrics(y_true, preds, model_name=None):
    '''
    INPUT:
    y_true - the y values that are actually true in the dataset (numpy array or pandas series)
    preds - the predictions for those values from some model (numpy array or pandas series)
    model_name - (str - optional) a name associated with the model if you would like to add it to the print statements

    OUTPUT:
    None - prints the accuracy, precision, recall, and F1 score
    '''
    if model_name == None:
        print('Accuracy score: ', format(accuracy_score(y_true, preds)))
        print('Precision score: ', format(precision_score(y_true, preds)))
        print('Recall score: ', format(recall_score(y_true, preds)))
        print('F1 score: ', format(f1_score(y_true, preds)))
        print('\n\n')

    else:
        print('Accuracy score for ' + model_name + ' :', format(accuracy_score(y_true, preds)))
        print('Precision score ' + model_name + ' :', format(precision_score(y_true, preds)))
        print('Recall score ' + model_name + ' :', format(recall_score(y_true, preds)))
        print('F1 score ' + model_name + ' :', format(f1_score(y_true, preds)))
        print('\n\n')


# Print Bagging scores
print_metrics(y_test, bag_preds, 'bagging')

# Print Random Forest scores
print_metrics(y_test, rf_preds, 'random forest')

# Print AdaBoost scores
print_metrics(y_test, ada_preds, 'adaboost')

# Naive Bayes Classifier scores
print_metrics(y_test, predictions, 'naive bayes')
