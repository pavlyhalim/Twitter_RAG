# Twitter Data Analysis with RAG and Prompt Engineering

This Python script leverages the power of Retrieval Augmented Generation (RAG) and prompt engineering to analyze Twitter data and answer your questions about NewJeans, a K-pop girl group.

## Features

- **RAG Implementation:** The script reads a JSON file containing Twitter data and employs RAG techniques to answer user-provided questions about the dataset.
- **Prompt Engineering:** It carefully crafts prompts for the OpenAI API to guide the model in understanding the data and generating relevant responses.
- **User-Friendly Interface:** A simple command-line interface makes it easy to ask questions and receive insightful answers.
- **Date Range Parsing:** Automatically detects and filters tweets based on date ranges mentioned in questions.
- **Relevance Scoring:** Uses TF-IDF and cosine similarity to retrieve the most relevant tweets for each question.
- **Token Counting:** Utilizes the `tiktoken` library to count tokens, ensuring responses stay within API limits.

## Requirements

- Python 3.7 or higher
- OpenAI API key
- Required Python libraries: `openai`, `sklearn`, `tiktoken`, `json`, `datetime`, `re`

## Installation

1. Clone the repository: `git clone https://github.com/your-username/your-repo-name.git`
2. Install required packages: `pip install openai scikit-learn tiktoken`
3. Ensure your OpenAI API key is set in the script:
   ```python
   openai.api_key = "your_openai_api_key_here"
   ```

## Usage

1. **Prepare your data:**
   - Place your Twitter data in a JSON file named `data.jsonl` in the same directory as the script.
   - Ensure the JSON file contains a `tweets` array and a `summary` object.

2. **Run the script:**
   - Execute the script: `python gpt-tweet.py`

3. **Ask questions:**
   - The script will prompt you to enter your question.
   - Type your question and press Enter.

4. **Get answers:**
   - The script will analyze the Twitter data and provide a comprehensive answer based on the information available in the JSON file.
   - It will list relevant tweets, provide analysis, and highlight key insights.

5. **Quit:**
   - Type 'quit' and press Enter to exit the program.

## Example

```
Enter your question (or 'quit'): What are the most recent 5 tweets about Hanni?
Answer:
[The script will provide a detailed answer, including the 5 most recent tweets mentioning Hanni, their URLs, and any relevant analysis.]
```

## How it Works

1. **Data Loading:** The script reads the Twitter data from the specified JSON file.
2. **Question Processing:** When you ask a question, the script parses it for date ranges and key terms.
3. **Tweet Retrieval:** Relevant tweets are retrieved based on TF-IDF similarity and keyword matching.
4. **RAG with OpenAI API:** A carefully crafted prompt, along with the relevant tweets and summary data, is sent to the OpenAI API.
5. **Answer Generation:** The OpenAI model processes the information and generates a comprehensive answer.
6. **Output:** The script presents the generated answer, including relevant tweets and analysis.

## Customization

- **File Path:** Modify the `file_path` variable to point to your Twitter data file if it's not in the default location.
- **OpenAI Model:** You can experiment with different OpenAI models by changing the `model` parameter in the `get_completion` function.
- **Maximum Tweets:** Adjust the `max_tweets` variable in the `main` function to control how many tweets are included in each analysis.
- **Prompt Optimization:** Further refine the prompt template in the `analyze_data` function to improve the accuracy and relevance of the answers for your specific use case.

## Disclaimer

This code utilizes the OpenAI API, and standard API usage charges apply. Make sure you understand the OpenAI API pricing before running this code. The script includes safeguards to prevent excessive token usage, but monitor your usage to avoid unexpected charges.
