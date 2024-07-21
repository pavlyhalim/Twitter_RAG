
# Twitter Data Analysis with RAG and Prompt Engineering

This Python script leverages the power of Retrieval Augmented Generation (RAG) and prompt engineering to analyze Twitter data and answer your questions.

## Features

- **RAG Implementation:** The script reads a CSV file containing Twitter data and employs RAG techniques to answer user-provided questions about the dataset.
- **Prompt Engineering:**  It carefully crafts prompts for the OpenAI API to guide the model in understanding the data and generating relevant responses.
- **User-Friendly Interface:** A simple command-line interface makes it easy to ask questions and receive insightful answers.

## Requirements

- Python 3.7 or higher
- OpenAI API key
- `openai` Python library
- `dotenv` Python library

## Installation

1. Clone the repository: `git clone https://github.com/your-username/your-repo-name.git`
2. Install required packages: `pip install openai python-dotenv`
3. Create a `.env` file in the project directory and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

## Usage

1.  **Prepare your data:**
    -   Place your Twitter data in a CSV file (e.g., `tweets.csv`).
    -   Make sure the file is in the same directory as the script or provide the correct file path in the code (`file_path` variable).
2.  **Run the script:**
    -   Execute the script: `python gpt-tweet.py`
3.  **Ask questions:**
    -   The script will prompt you to enter your question.
    -   Type your question and press Enter.
4.  **Get answers:**
    -   The script will analyze the Twitter data and provide an answer based on the information available in the CSV.
    -   If the answer cannot be found in the data, it will let you know.
5.  **Quit:**
    -   Type 'quit' and press Enter to exit the program.

## Example

```
Enter your question (or 'quit' to exit): What are the most common hashtags used in this dataset?

Answer:
#DataScience #AI #MachineLearning #Python
```

## How it Works

1.  **Data Loading:** The script reads the Twitter data from the specified CSV file.
2.  **Question Processing:** When you ask a question, the script embeds it within a carefully designed prompt.
3.  **RAG with OpenAI API:** The prompt, along with the relevant Twitter data, is sent to the OpenAI API (using `gpt-3.5-turbo` by default) (This model's maximum context length is 16385 tokens).
4.  **Answer Generation:** The OpenAI model processes the information and generates an answer based on its understanding of the data and the question.
5.  **Output:** The script presents the generated answer to you.

## Customization

-   **File Path:** Modify the `file_path` variable to point to your Twitter data file.
-   **OpenAI Model:** You can experiment with different OpenAI models by changing the `model` parameter in the `get_completion` function.
-   **Prompt Optimization:**  Further refine the prompt structure to improve the accuracy and relevance of the answers for your specific use case.

## Disclaimer

This code utilizes the OpenAI API, and standard API usage charges apply. Make sure you understand the OpenAI API pricing before running this code.


**Remember to replace placeholders like `your_openai_api_key_here`, `tweets.csv` with your actual details.**
