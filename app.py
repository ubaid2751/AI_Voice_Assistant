import os
from datetime import datetime
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

template = """
You are a highly intelligent, knowledgeable, and empathetic assistant designed to provide accurate and helpful answers to user queries.

Here is the conversation so far:
====================
{convo}
====================

Based on the context above, answer the following question:
====================
Question: {question}
====================

Instructions for responses:
1. If the query involves generating or writing code, respond **only** in the following format:
    filename: "<desired filename>"
    code: "<generated code>"

    The `filename` is the name of the file you want to create, and `code` should be the generated code (as a string).
2. For other types of queries, provide a concise response based on the question and context.

Instructions for general queries:
1. Provide a concise and accurate response based on the question and context.
2. Maintain a professional tone and ensure clarity in the explanation.
3. If the question is unclear or incomplete, ask for clarification.
4. Offer additional examples or suggestions if it helps in better understanding.

Your response:
"""

prompt = ChatPromptTemplate.from_template(template)
model = OllamaLLM(model="llama3.2")
chain = prompt | model

def save_response(response, directory="tmp"):
    os.makedirs(directory, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(directory, f"response_{timestamp}.txt")
    
    lst = os.listdir(directory)
    num_files = len(lst)
    if num_files >= 10:
        for file in lst:
            os.remove(os.path.join(directory, file))
    
    with open(filename, "w", encoding="utf-8") as file:
        file.write(response)
    print(f"Response saved to {filename}")

def rnw_code(response):
    try:
        lines = response.strip().splitlines()
        filename_line = lines[0].split(":")[1].strip().strip('"')
        code_lines = "\n".join(lines[1:]).strip()
        with open(filename_line, "w", encoding="utf-8") as code_file:
            code_file.write(code_lines)
        print(f"Code successfully written to {filename_line}")
    except Exception as e:
        print(f"Error processing the custom format: {e}")

def handle_conv():
    convo = []
    context = []
    MAX_CONTEXT_LENGTH = 10
    print("Welcome to the chatbot! Type 'exit' to end the conversation.")
    while True:
        user_input = input(f"{bcolors.OKCYAN}You: ")
        convo.append({'role': 'user', 'content': user_input})
        print(f"{bcolors.ENDC}", end="")
        if user_input.lower() == "exit":
            break
    
        response = ""
        print(f"{bcolors.OKGREEN}BOT: {bcolors.ENDC}", end="")
        for ans in chain.stream({
            # "context": context,
            "convo": "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in convo]),
            "question": user_input,
        }):
            print(f"{bcolors.OKGREEN}{ans}", end="")
            response += ans
        
        convo.append({'role': 'bot', 'content': response})
        print(f"{bcolors.ENDC}")
        if response.strip().startswith("filename:"):
            save_response(response)
            rnw_code(response)
        context.append({"User": user_input, "Bot": response})
        if len(context) > MAX_CONTEXT_LENGTH:
            context = context[-MAX_CONTEXT_LENGTH:]

handle_conv()
print(f"{bcolors.ENDC}")