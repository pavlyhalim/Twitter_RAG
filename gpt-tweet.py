import json
import re
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import faiss
from langdetect import detect
from konlpy.tag import Okt
import openai
import tiktoken
from tqdm import tqdm
import multiprocessing as mp
import pytz
import logging
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
model = AutoModel.from_pretrained("distilbert-base-multilingual-cased")
okt = Okt()

openai.api_key = "YOUR-KEY"

gpt4_tokenizer = tiktoken.encoding_for_model("gpt-4o")

processed_df = None
faiss_index = None

def check_authentication():
    try:
        openai.Model.list()
        logging.info("Authentication successful!")
        return True
    except openai.error.AuthenticationError:
        logging.error("Authentication failed. Please check your API key.")
        return False

def load_data(file_path: str) -> pd.DataFrame:
    tweets = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file, desc="Loading tweets"):
            try:
                tweet = json.loads(line.strip())
                tweet_id = tweet.get('id_str', '')
                tweet['url'] = f"https://twitter.com/i/web/status/{tweet_id}" if tweet_id else ''
                tweets.append(tweet)
            except json.JSONDecodeError:
                logging.warning(f"Skipping invalid JSON line: {line}")
    return pd.DataFrame(tweets)

def preprocess_text(text: str) -> str:
    text = re.sub(r'http\S+|@\w+|#\w+', '', text)
    text = text.lower().strip()
    return text

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except:
        return 'en' 

def preprocess_korean(text: str) -> str:
    tokens = okt.morphs(text, stem=True)
    return ' '.join(tokens)

@torch.no_grad()
def encode_text(text: str) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()[0]

def process_chunk(chunk):
    chunk['preprocessed_text'] = chunk['text'].apply(preprocess_text)
    chunk['language'] = chunk['preprocessed_text'].apply(detect_language)
    chunk.loc[chunk['language'] == 'ko', 'preprocessed_text'] = chunk.loc[chunk['language'] == 'ko', 'preprocessed_text'].apply(preprocess_korean)
    chunk['embedding'] = chunk['preprocessed_text'].apply(encode_text)
    return chunk

def parallel_process_tweets(df: pd.DataFrame, num_processes: int = mp.cpu_count()) -> pd.DataFrame:
    chunks = np.array_split(df, num_processes)
    with mp.get_context("spawn").Pool(num_processes) as pool:
        processed_chunks = list(tqdm(pool.imap(process_chunk, chunks), total=len(chunks), desc="Processing tweets"))
    return pd.concat(processed_chunks)

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index

def retrieve_relevant_tweets(query: str, k: int = 100) -> pd.DataFrame:
    global processed_df, faiss_index
    
    query_embedding = encode_text(preprocess_text(query))
    
    D, I = faiss_index.search(query_embedding.reshape(1, -1), k)
    relevant_tweets = processed_df.iloc[I[0]]
    
    current_date = pd.Timestamp.now(pytz.utc)
    relevant_tweets['created_at'] = pd.to_datetime(relevant_tweets['created_at'], utc=True)
    relevant_tweets['days_old'] = (current_date - relevant_tweets['created_at']).dt.total_seconds() / (24 * 3600)
    relevant_tweets['time_decay'] = np.exp(-relevant_tweets['days_old'] / 7)
    
    relevant_tweets['relevance_score'] = D[0] * relevant_tweets['time_decay']
    relevant_tweets = relevant_tweets.sort_values('relevance_score', ascending=False)
    
    return relevant_tweets

def format_tweets_for_gpt(tweets: pd.DataFrame, max_tokens: int = 3500) -> str:
    formatted_tweets = []
    total_tokens = 0
    
    for idx, tweet in tweets.iterrows():
        formatted_tweet = f"Tweet {idx + 1}:\n"
        formatted_tweet += f"Text: {tweet['text']}\n"
        formatted_tweet += f"Date: {tweet['created_at']}\n"
        formatted_tweet += f"Likes: {tweet.get('favorite_count', 'N/A')}\n"
        formatted_tweet += f"Retweets: {tweet.get('retweet_count', 'N/A')}\n"
        formatted_tweet += f"URL: {tweet['url']}\n\n"
        
        tweet_tokens = len(gpt4_tokenizer.encode(formatted_tweet))
        if total_tokens + tweet_tokens > max_tokens:
            break
        
        formatted_tweets.append(formatted_tweet)
        total_tokens += tweet_tokens
    
    additional_info = f"\n\nNote: {len(tweets) - len(formatted_tweets)} additional relevant tweets were found but not included due to token limits."
    
    return "".join(formatted_tweets) + additional_info

def generate_gpt4_analysis(query: str, formatted_tweets: str, total_relevant_tweets: int) -> str:
    prompt = f"""Analyze the following tweets about NewJeans in response to this specific query: "{query}"

{formatted_tweets}

Total number of relevant tweets found: {total_relevant_tweets}

Please provide a focused analysis that directly addresses the query. Include:
1. A concise summary of the main points in these tweets that are directly related to the query.
2. The overall sentiment towards NewJeans specifically regarding the topic in the query.
3. Notable opinions or reactions from fans that are relevant to the query.
4. Mentions of specific songs, performances, or activities that relate to the query, if any.
5. Engagement levels (likes, retweets) for tweets most relevant to the query and what they might indicate.
6. Any other insights you can draw from these tweets that are directly relevant to the query.
7. Reference specific tweets by their number and include their URLs when discussing particular points.

Your analysis should be detailed, insightful, and laser-focused on addressing the query. If the tweets don't contain information relevant to the query, please mention this and explain why. Concentrate only on information that is directly related to the query."""

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert analyst specializing in K-pop and social media trends, with a focus on NewJeans. Your task is to provide query-specific analysis of tweets, including relevant URLs."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        return response.choices[0].message['content']
    except Exception as e:
        logging.error(f"Error in GPT-4 analysis: {str(e)}")
        return "An error occurred during the GPT-4 analysis. Please try again."

def preprocess_and_index_tweets(file_path: str):
    global processed_df, faiss_index
    
    logging.info("Loading data...")
    df = load_data(file_path)
    
    logging.info("Processing tweets...")
    processed_df = parallel_process_tweets(df)
    
    processed_df['created_at'] = pd.to_datetime(processed_df['created_at'], utc=True)
    
    logging.info("Building FAISS index...")
    embeddings = np.vstack(processed_df['embedding'].values)
    faiss_index = build_faiss_index(embeddings)
    
    logging.info("Preprocessing and indexing complete.")

def analyze_query(query: str) -> str:
    try:
        logging.info(f"Analyzing query: {query}")
        
        logging.info("Retrieving relevant tweets...")
        relevant_tweets = retrieve_relevant_tweets(query)
        
        logging.info("Formatting tweets for GPT-4...")
        formatted_tweets = format_tweets_for_gpt(relevant_tweets)
        
        logging.info("Generating GPT-4 analysis...")
        analysis = generate_gpt4_analysis(query, formatted_tweets, len(relevant_tweets))
        
        logging.info(f"Total relevant tweets found: {len(relevant_tweets)}")
        logging.info("GPT-4 Analysis complete.")
        
        return analysis
    except Exception as e:
        logging.error(f"An error occurred during query analysis: {str(e)}")
        return "An error occurred during the analysis. Please try again."

def main(file_path: str):
    if not check_authentication():
        return "Authentication failed. Please check your API key."

    try:
        preprocess_and_index_tweets(file_path)
        
        while True:
            query = input("Enter your specific question about NewJeans (or type 'exit' to quit): ")
            if query.lower() == 'exit':
                break
            analysis = analyze_query(query)
            print("\nAnalysis:")
            print(analysis)
            print("\n" + "="*50 + "\n")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {str(e)}")
        print("An unexpected error occurred. Please check the logs for more information.")

if __name__ == "__main__":
    file_path = "recent_7k_not_jeans.jsonl"
    main(file_path)