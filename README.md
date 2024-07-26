# Tweet Analysis System

## Overview

This project is an advanced tweet analysis system specifically designed for analyzing tweets related to the K-pop group NewJeans. It uses natural language processing, machine learning, and the GPT-4 API to provide in-depth, query-specific analyses of tweets about NewJeans.

## Features

- Load and preprocess tweets from a JSONL file
- Multilingual support with automatic language detection
- Advanced text preprocessing including Korean language support
- Parallel processing for efficient tweet analysis
- FAISS-based similarity search for relevant tweet retrieval
- Time-aware relevance scoring
- GPT-4 powered analysis of retrieved tweets
- Interactive query system for user-specific questions

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- FAISS
- Pandas
- NumPy
- Langdetect
- KoNLPy
- OpenAI API
- Tiktoken
- Tqdm
- Pytz

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/your-username/newjeans-tweet-analysis.git
   cd newjeans-tweet-analysis
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your OpenAI API key:
   - Create a `.env` file in the project root
   - Add your OpenAI API key to the file:
     ```
     OPENAI_API_KEY=your-api-key-here
     ```

4. Prepare your tweet data:
   - Ensure your tweet data is in JSONL format
   - Update the `file_path` variable in the script with the path to your JSONL file

## Usage

1. Run the script:
   ```
   python gpt-tweet.py
   ```

2. Enter your specific questions about NewJeans when prompted.

3. Review the analysis provided by the system.

4. Type 'exit' when you're done asking questions.

## How It Works

1. **Data Loading**: The system loads tweets from a JSONL file.
2. **Preprocessing**: Tweets are preprocessed, including text cleaning and language detection.
3. **Embedding**: Each tweet is converted into a vector representation using a pre-trained language model.
4. **Indexing**: A FAISS index is built for efficient similarity search.
5. **Query Processing**: User queries are processed to find relevant tweets.
6. **Analysis**: GPT-4 analyzes the relevant tweets and provides insights based on the user's query.

## Limitations

- The system requires a significant amount of RAM, especially for large tweet datasets.
- Analysis quality depends on the availability and relevance of tweets in the dataset.
- The OpenAI API key is required for GPT-4 analysis.

## Contributing

Contributions to improve the system are welcome. Please feel free to submit issues or pull requests.
