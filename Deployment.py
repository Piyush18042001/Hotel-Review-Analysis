import streamlit as st
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

st.title('Hotel Review Classifier')

user_input=st.text_area("Enter a review:")

data_re=pd.read_excel('hotel_reviews (1).xlsx')

data_neg = data_re.loc[data_re["Rating"]<3]
data_neg = data_neg.reset_index(drop = True)
data_five = data_re.loc[data_re['Rating'] == 5]
data_five = data_five.reset_index(drop = True)
data_pos = data_five.loc[:len(data_neg)]
data_all = pd.concat([data_neg,data_pos], axis = 0)
data_all = data_all.reset_index(drop = True)

data_all["Sentiment"]=np.where(data_all["Rating"] == 5, "Positive" , "Negative")
data_all= data_all.sample(frac=1)
data_all= data_all.reset_index(drop = True)

x=data_all.Review
y=data_all.Sentiment

v=CountVectorizer()
x_vec = v.fit_transform(x)

nb_classifier = MultinomialNB()

# Sentiment prediction
if st.button("Predict Sentiment"):
    # Preprocess user input
    user_input_vec=v.transform([user_input])
    nb_classifier.fit(x_vec, y)
    user_sentiment=nb_classifier.predict(user_input_vec)

    # Display sentiment
    st.write(f"Predicted Sentiment: {user_sentiment}")

