import pandas as pd
import streamlit as st
import plotly.express as px

# Title
st.title("Sentiment Analysis Dashboard for Valuation Methodology Survey")

# Load data from Google Sheet as CSV
sheet_url = "https://docs.google.com/spreadsheets/d/1uIKALZidjfcFRd2ph9fDch4-upR4QQxf30i70H4gtOU/export?format=csv"
df = pd.read_csv(sheet_url)

# Rename or alias columns for use
df['Education Level'] = df['3.       What is your highest level of education?']
df['Country'] = df['1.       In which country are you registered to practice?']
df['Sentiment'] = df['vader_sentiment']

# Sidebar Filter
group_col = st.sidebar.selectbox("Group sentiment by:", ['Education Level', 'Country'])

# Bar Chart
grouped = df.groupby([group_col, 'Sentiment']).size().reset_index(name='Count')
fig_bar = px.bar(grouped, x=group_col, y='Count', color='Sentiment',
                 title=f"Sentiment by {group_col}", barmode='group')
st.plotly_chart(fig_bar, use_container_width=True)

# Pie Chart
sentiment_counts = df['Sentiment'].value_counts().reset_index()
sentiment_counts.columns = ['Sentiment', 'Count']
fig_pie = px.pie(sentiment_counts, names='Sentiment', values='Count',
                 title="Overall Sentiment Distribution")
st.plotly_chart(fig_pie, use_container_width=True)

# Show raw data
if st.checkbox("Show raw data"):
    st.write(df[['Country', 'Education Level', 'Sentiment',
                 '8.       What steps, in your opinion, can be taken to promote the adoption of advanced valuation methods in your country’s valuations?',
                 '9.       Do you have any additional comments on advanced valuation methodologies and their adoption in your country of practice?']])
