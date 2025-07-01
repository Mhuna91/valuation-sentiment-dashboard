import pandas as pd
import streamlit as st
from textblob import TextBlob
import nltk
nltk.download('punkt')

# Load Google Sheet
sheet_url = "https://docs.google.com/spreadsheets/d/1uIKALZidjfcFRd2ph9fDch4-upR4QQxf30i70H4gtOU/export?format=csv"
df = pd.read_csv(sheet_url)

# Apply sentiment analysis to Q8 and Q9
def get_sentiment(text):
    if pd.isna(text):
        return 0
    return TextBlob(str(text)).sentiment.polarity

df['Q8_Sentiment'] = df['8.       What steps, in your opinion, can be taken to promote the adoption of advanced valuation methods in your country’s valuations?'].apply(get_sentiment)
df['Q9_Sentiment'] = df['9.       Do you have any additional comments on advanced valuation methodologies and their adoption in your country of practice?'].apply(get_sentiment)

# Average sentiment score
df['Sentiment'] = df[['Q8_Sentiment', 'Q9_Sentiment']].mean(axis=1)

# Show dashboard
st.title("📊 Sentiment Analysis Dashboard for Valuation Methodology Survey")

st.write("### Average Sentiment Score by Education Level")
st.bar_chart(df.groupby("3.       What is your highest level of education?")['Sentiment'].mean())

st.write("### Average Sentiment Score by Country")
st.bar_chart(df.groupby("1.       In which country are you registered to practice?")['Sentiment'].mean())

st.write("### Raw Data Preview")
st.dataframe(df[['1.       In which country are you registered to practice?', 
                 '3.       What is your highest level of education?',
                 '8.       What steps, in your opinion, can be taken to promote the adoption of advanced valuation methods in your country’s valuations?',
                 '9.       Do you have any additional comments on advanced valuation methodologies and their adoption in your country of practice?',
                 'Sentiment']])
