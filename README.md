# YouTube Video Summary
YouTube being one of the major source of knowledge and information, spending time to watch long videos can be painful in terms of time and patience. This project aims to provide a summary of any Youtube video by using two simple APIs.

## Solution Statement
Advanced Natural Language Processing techinques are used in this project to generate the summary using Streamlit webapp and Tkinter GUI. The methods used are Tf-Idf and Genism. Tf-Idf method involves ranking the scentences based on the the words frequency based vectorization. Genism method involves ranking the scentences based on the embeddings based vectorization of words. After ranking the scentences, based on the fraction input, the top scentences are selected to form the summary of the YouTube video long transcripts.

## Model Evaluation
The above extractive text summarization methods are evaluated based on the ROUGE score. ROUGE score was exclusively developed to become the standard evaluation measure for summarization tasks.The methods are tested on the long document summarization dataset "govreport-summarization" from Hugging Face. It works based on the precision, recall and F1-score of overlapping words between the generated summary and refernce summary. ROUGE score ranges between 0 to 1.
Below are the results.

### ROUGE Score
|Method|ROUGE-1_F-Score|
|-|-|
|Tf-Idf|0.3613|
|Genism|0.4064|

## Working
### Tkinter GUI
The "youtube_video_summarizer.ipynb" can be run to generate a tkinter GUI. Here you can paste the link, select a method for summarization, enter a desired fraction value to generate the summary and provide the desired location to save the entire transcript and the summary.

### Streamlit Webapp
The "app.py" file can be run to generate a streamlit webapp. The process remains the same like tkinter GUI, paste the link, select a method for summarization and a desired fraction value to display the summary.

```
streamlit run app.py
```

## Future Work
In this project, extractive text summarization methods are used to summarize the YouTube video transcripts which invloves using the top ranked scentences as it is. As a part of future work, abstractive text summarization method can be implemented which involves a complex process like understanding the language, the context and generating new sentences. This frees the model of the constraint of using pre-written text but involves using large-scale data during training.
ROUGE score can not be used for evaluating abstractive summarization methods since new words/sentences are used to summarize the transcript. Instead, metrics such as Bert-score can be used to evaluate abstractive summarization methods. It focuses on computing the semantic similarity between the tokens of generated summary and reference summary. 
