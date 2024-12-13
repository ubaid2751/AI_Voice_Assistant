import os
from datetime import datetime
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

template = """
You are a highly intelligent, knowledgeable, and empathetic assistant designed to provide accurate and helpful answers to user queries.

Here is the context of the conversation so far:
====================
{context}
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

Additional features:
1. Manage conversational context, ensuring only the last 10 exchanges are retained for relevance.
2. Save responses to timestamped files for record-keeping.
3. Automatically write the code snippet from the response to a file using the specified filename.

Your response:
"""

prompt = ChatPromptTemplate.from_template(template)
model = OllamaLLM(model="llama3.2")
chain = prompt | model

def save_custom_response(response, directory="tmp"):
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
    return filename

def read_and_write_code(response):
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
    context = []
    MAX_CONTEXT_LENGTH = 10
    print("Welcome to the chatbot! Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        formatted_context = "\n".join([f"User: {item['User']}\nBot: {item['Bot']}" for item in context])
        response = ""
        for ans in chain.stream({
            "context": formatted_context,
            "question": user_input
        }):
            print(ans, end="")
            response += ans
        print()
        if response.strip().startswith("filename:"):
            response_file = save_custom_response(response)
            read_and_write_code(response)
        context.append({"User": user_input, "Bot": response})
        if len(context) > MAX_CONTEXT_LENGTH:
            context = context[-MAX_CONTEXT_LENGTH:]

handle_conv()
