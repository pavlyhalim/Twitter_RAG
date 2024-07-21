import os
import openai
from dotenv import load_dotenv

load_dotenv()

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

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


def analyze_data(file_content, question):
    prompt = f"""
    You are an AI assistant tasked with analyzing Twitter data and answering questions about it.
    The data is provided below, enclosed in triple backticks. Your task is to analyze this data
    and answer the following question:

    {question}

    Please base your answer only on the information provided in the data. If the answer cannot be
    found in the data, say so. When referencing tweets, please include the tweet URL.

    Here's the data:

    ```
    {file_content}
    ```

    Now, please answer the question based on this data.
    """

    return get_completion(prompt)

def main():
    file_path = "1k_not_jeans_less.csv" 
    
    try:
        file_content = read_file(file_path)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found. Please make sure the file exists and the path is correct.")
        return
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return

    while True:
        question = input("Enter your question (or 'quit' to exit): ")
        if question.lower() == 'quit':
            break

        try:
            answer = analyze_data(file_content, question)
            print("\nAnswer:")
            print(answer)
        except ValueError as e:
            print(f"\nError: {str(e)}")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {str(e)}")
        
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()