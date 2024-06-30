# TODO: 2.3: Implementing Bag of Words in scikit-learn
documents = ['Hello, how are you!',
             'Win money, win from home.',
             'Call me now.',
             'Hello, Call hello you tomorrow?']

from sklearn.feature_extraction.text import CountVectorizer

count_vector = CountVectorizer()

count_vector.fit(documents)
print(count_vector.get_feature_names())

doc_array = count_vector.transform(documents).toarray()
print(doc_array)

import pandas as pd

frequency_matrix = pd.DataFrame(doc_array,
                                columns=count_vector.get_feature_names())
print(frequency_matrix)
