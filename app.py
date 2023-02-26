### All imports

import youtube_transcript_api
import os
import nltk
import re
import sklearn
import numpy as np
import streamlit as st

from youtube_transcript_api import YouTubeTranscriptApi
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
from gensim.summarization import summarize

import warnings
warnings.filterwarnings("ignore")

# Fetch Text From Url
@st.cache(allow_output_mutation=True)
def get_transcript(url):
    """
    PARAMETER
    url: Valid YouTube link with valid transcipt.
    
    RETURN
    transcript: String of whole video transcript
    sentences: Transcript divided as list of sentence tokens.
    
    """
    unique_id = url.split("=")[1:]
    unique_id = "=".join([x for x in unique_id])
    try:
        sub = YouTubeTranscriptApi.get_transcript(unique_id)
        transcript = " ".join([x['text'] for x in sub])

        transcript = transcript.replace("\n","")
        sentences = sent_tokenize(transcript)
    except:
        print("Try with a valid YouTube URL")
    return transcript, sentences



@st.cache(allow_output_mutation=True)
def tf_idf_based_summary(sentences, fraction):
    """
    PARAMETER
    sentences: Transcript divided as list of sentence tokens.
    fraction: Decimal value of desired length of the summary.
    
    RETURN
    tf_idf_summary: Summary generated based on the Tf-Idf method.
    
    """
    organized_sent = {k:v for v,k in enumerate(sentences)}
    tf_idf = TfidfVectorizer(min_df=2, 
                                    strip_accents='unicode',
                                    max_features=None,
                                    lowercase = True,
                                    token_pattern=r'w{1,}',
                                    ngram_range=(1, 3), 
                                    use_idf=1,
                                    smooth_idf=1,
                                    sublinear_tf=1,
                                    stop_words = 'english')
    sentence_vectors = tf_idf.fit_transform(sentences)
    sent_scores = np.array(sentence_vectors.sum(axis=1)).ravel()
    num_sent=int(np.ceil(len(sentences)*fraction))
    top_n_sentences = [sentences[index] for index in np.argsort(sent_scores, axis=0)[::-1][:num_sent]]
    # mapping the scored sentences with their indexes as in the subtitle
    mapped_sentences = [(sentence,organized_sent[sentence]) for sentence in top_n_sentences]
    # Ordering the top-n sentences in their original order
    mapped_sentences = sorted(mapped_sentences, key = lambda x: x[1])
    ordered_sentences = [element[0] for element in mapped_sentences]
    # joining the ordered sentence
    tf_idf_summary = " ".join(ordered_sentences)

    return tf_idf_summary

def main():
	"""Summarizer Streamlit App"""

	st.title("Youtube video summarizer")

	method = ["TfIdf","Genism"]
	option = st.sidebar.selectbox("Select Method",method)
	url = st.sidebar.text_input("Enter a YouTube URL", "")
	fraction = st.sidebar.text_input("Enter a fraction", "")
	fraction = float(fraction)
	try:
		transcript, sentences=get_transcript(url)
		if len(sentences)>1:
			if option == "TfIdf":
				summary=tf_idf_based_summary(sentences, fraction)
			elif option == "Genism":
				summary=summarize(text=transcript, ratio=fraction, split=False).replace("\n", " ")
			st.subheader("SUMMARY TEXT")
			st.write(summary)
			st.write("Original text length", len(transcript))
			st.write("Summary text length", len(summary))
	except:
		st.write("URL\\Transcript invalid")

if __name__ == '__main__':
	main()
