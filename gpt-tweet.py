import json
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
import re
import tiktoken

openai.api_key = "YOUR_KEY"

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def get_completion(prompt, model="gpt-3.5-turbo"):
    if not openai.api_key:
        raise ValueError("No OpenAI API key found. Please check the API key in the script.")
    
    messages = [{"role": "user", "content": prompt}]
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message["content"]
    except openai.error.AuthenticationError:
        raise ValueError("Invalid OpenAI API key. Please check the API key in the script.")
    except Exception as e:
        raise ValueError(f"An error occurred while calling the OpenAI API: {str(e)}")

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def parse_date_range(question):
    start_date = None
    end_date = None
    time_period_match = re.search(r'last (\d+) (day|week|month)s?', question.lower())
    if time_period_match:
        num = int(time_period_match.group(1))
        unit = time_period_match.group(2)
        if unit == 'day':
            start_date = datetime.now() - timedelta(days=num)
        elif unit == 'week':
            start_date = datetime.now() - timedelta(weeks=num)
        elif unit == 'month':
            start_date = datetime.now() - timedelta(days=num*30)
        end_date = datetime.now()
    else:
        date_matches = re.findall(r'(\d{4}-\d{2}-\d{2})', question)
        if len(date_matches) == 2:
            start_date = datetime.strptime(date_matches[0], '%Y-%m-%d')
            end_date = datetime.strptime(date_matches[1], '%Y-%m-%d')
        elif len(date_matches) == 1:
            if 'from' in question.lower() or 'since' in question.lower() or 'after' in question.lower():
                start_date = datetime.strptime(date_matches[0], '%Y-%m-%d')
            elif 'until' in question.lower() or 'before' in question.lower():
                end_date = datetime.strptime(date_matches[0], '%Y-%m-%d')
    return start_date, end_date

def retrieve_relevant_tweets(data, question, max_tweets=50, sort_by_date=False):
    tweets = data['tweets']
    
    if sort_by_date:
        tweets = sorted(tweets, key=lambda x: datetime.strptime(x['created_at'], '%Y-%m-%dT%H:%M:%S.%fZ'), reverse=True)
        return tweets[:max_tweets]
    
    vectorizer = TfidfVectorizer()
    tweet_texts = [tweet['text'] for tweet in tweets]
    tweet_vectors = vectorizer.fit_transform(tweet_texts)
    
    question_vector = vectorizer.transform([question])
    similarities = cosine_similarity(question_vector, tweet_vectors)
    
    sorted_indices = similarities.argsort()[0][::-1]
    
    relevant_tweets = []
    unique_tweet_texts = set()
    
    key_terms = re.findall(r'\b\w+\b', question.lower())
    
    def tweet_matches_query(tweet):
        tweet_text = tweet['text'].lower()
        tweet_data = json.dumps(tweet).lower()
        
        return any(term in tweet_text or term in tweet_data for term in key_terms)
    
    for idx in sorted_indices:
        tweet = tweets[idx]
        if tweet_matches_query(tweet) and tweet['text'] not in unique_tweet_texts:
            relevant_tweets.append(tweet)
            unique_tweet_texts.add(tweet['text'])
            if len(relevant_tweets) >= max_tweets:
                break
    
    return relevant_tweets

def analyze_data(data, relevant_tweets, question):
    summary = json.dumps(data['summary'], indent=2)
    
    total_tweets = len(relevant_tweets)
    
    if total_tweets == 0:
        return "No relevant tweets found for this query."
    
    prompt_template = """
    You are an expert analyst of Twitter data for NewJeans, a K-pop girl group. Analyze the provided data and answer the following question comprehensively:

    {question}

    Follow these guidelines strictly:

    1. Begin with a concise summary (2-3 sentences) of the key insights directly related to the question.
    2. List ALL relevant tweets, up to the maximum allowed. For each tweet, provide:
       a. A consecutive number
       b. The full tweet text
       c. The tweet's URL
       d. Any relevant metadata (e.g., engagement metrics, timestamp) if available and pertinent
    3. If the question asks for specific information (e.g., number of tweets, trends, statistics):
       a. Provide the exact information requested
       b. If possible, include brief analysis or context for the information
    4. For questions about music or performances:
       a. Note any mentions of song titles, music shows, or performance venues
       b. Highlight any fan reactions or engagement related to music content
    5. For member-specific questions:
       a. Focus on tweets that mention the specific member(s)
       b. Highlight any unique activities or characteristics mentioned for that member
    6. Identify and briefly explain any notable trends or patterns in the tweets
    7. If there are tweets in languages other than English, provide brief translations of key points
    8. Do not repeat information unnecessarily
    9. If any part of the question cannot be answered with the given data, explicitly state this
    10. Conclude by stating the total number of relevant tweets found and how many were included in the response

    Use this summary data for context:
    {summary}

    Analyze these relevant tweets:
    {tweets}

    Provide your comprehensive analysis and answer based on the above guidelines.
    """
    
    tweets = json.dumps(relevant_tweets, indent=2)
    
    prompt = prompt_template.format(
        question=question,
        summary=summary,
        tweets=tweets
    )
    
    response = get_completion(prompt)
    return response

def main():
    file_path = "processed_for_qa.jsonl"
    
    try:
        data = load_data(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found. Please make sure the file exists and the path is correct.")
        return
    except json.JSONDecodeError:
        print(f"Error: File '{file_path}' is not a valid JSON file.")
        return
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return

    print(f"Successfully loaded data with {data['summary']['total_tweets']} tweets.")

    while True:
        question = input("Enter your question (or 'quit'): ")
        if question.lower() == 'quit':
            break

        try:
            max_tweets = 40
            sort_by_date = False
            if "show me the most recent" in question.lower() or "what are the most recent" in question.lower():
                match = re.search(r'(show me|what are) the most recent (\d+)', question.lower())
                if match:
                    max_tweets = int(match.group(2))
                sort_by_date = True
                question = "What are the most recent tweets?"

            relevant_tweets = retrieve_relevant_tweets(data, question, max_tweets=max_tweets, sort_by_date=sort_by_date)
            answer = analyze_data(data, relevant_tweets, question)
            print("\nAnswer:")
            print(answer)
        except ValueError as e:
            print(f"\nError: {str(e)}")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {str(e)}")
        
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()
