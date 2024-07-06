import streamlit as st
import re
import json
import fitz
import pickle
import pandas as pd
import matplotlib.pyplot as plt

def ConvString(talklist):
    RetString = ""
    for line in talklist:
        x = line.split(' ', 4)[-1]
        x = re.sub(r'\[.*\]', '', x)
        x = re.sub('_1', '', x)
        RetString += x + ' '
    RetString = re.sub(r'\s+', ' ', RetString)
    return RetString

def process_resume(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    tx = " ".join(text.split('\n'))
    return tx

def main():
    st.title("Resume Scanner")

    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        with st.spinner('Processing...'):
            tx = process_resume(uploaded_file)

            tfidf = pickle.load(open('tfidf.pickle', 'rb'))
            clf = pickle.load(open('model_pkl', 'rb'))

            features = tfidf.transform([tx]).toarray()
            results = (clf.predict_proba(features))

            res = dict(zip(clf.classes_, results[0]))

            # Convert the results to a pandas DataFrame
            df = pd.DataFrame(list(res.items()), columns=['Job Role', 'Probability'])
            df['Probability'] = df['Probability'] * 100  # Convert to percentage
            df = df.sort_values(by='Probability', ascending=False).head(10)  # Get top 10 results

            # Plotting the bar graph
            plt.figure(figsize=(10, 6))
            bars = plt.barh(df['Job Role'], df['Probability'], color='skyblue')
            plt.xlabel('Probability (%)')
            plt.title('Top 10 Job Role Probabilities')
            plt.gca().invert_yaxis()  # Invert y-axis to show the highest probability at the top

            # Adding percentage labels
            for bar in bars:
                plt.text(
                    bar.get_width(),
                    bar.get_y() + bar.get_height() / 2,
                    f'{bar.get_width():.2f}%',
                    va='center',
                    ha='left',
                    fontsize=10,
                    color='black'
                )

            st.pyplot(plt)  # Display the plot in Streamlit

if __name__ == "__main__":
    main()
