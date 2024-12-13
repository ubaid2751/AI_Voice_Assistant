import yaml
import os
from datetime import datetime
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

ChatPromptTemplate

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
1. If the query involves generating or writing code, respond **only** in YAML format.
2. Generate a well-structured YAML format with the following keys:
    - "text": A concise explanation describing the purpose and functionality of the code.
    - "code": The actual code snippet, included as a block literal (using the `|` syntax).
    - "language_extension": The file extension for the code (e.g., ".py" for Python, ".cpp" for C++).
    - "filename": The desired name of the file (without the extension).

The YAML response should adhere to this structure:
text: "This script demonstrates the requested functionality."
code: |
  def add(a, b):
      return a + b

  print(add(5, 3))
language_extension: ".py"
filename: "example"

Instructions for general queries:
1. Provide a concise and accurate response based on the question and context.
2. Maintain a professional tone and ensure clarity in the explanation.
3. If the question is unclear or incomplete, ask for clarification.
4. Offer additional examples or suggestions if it helps in better understanding.

Additional features:
1. Manage conversational context, ensuring only the last 10 exchanges are retained for relevance.
2. Save YAML responses to timestamped files for record-keeping.
3. Automatically write the code snippet from the YAML response to a file using the specified filename and extension.

Your response:
"""

prompt = ChatPromptTemplate.from_template(template)
model = OllamaLLM(model="llama3.2:latest")
chain = prompt | model

def save_yaml_response(response, directory="tmp"):
    """Save the YAML response to a file."""
    os.makedirs(directory, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(directory, f"response_{timestamp}.yaml")

    with open(filename, "w", encoding="utf-8") as file:
        file.write(response)

    print(f"YAML response saved to {filename}")
    return filename

def read_and_write_code(yaml_file):
    try:
        with open(yaml_file, "r", encoding="utf-8") as file:
            yaml_obj = yaml.safe_load(file)

        filename = f"{yaml_obj['filename']}{yaml_obj['language_extension']}"
        with open(filename, "w", encoding="utf-8") as code_file:
            code_file.write(yaml_obj['code'])

        print(f"Code successfully written to {filename}")
    except (yaml.YAMLError, KeyError) as e:
        print(f"Error processing YAML: {e}")

def handle_conv():
    context = []
    MAX_CONTEXT_LENGTH = 10

    print("Welcome to the chatbot! Type 'exit' to end the conversation.")

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        formatted_context = "\n".join(
            [f"User: {item['User']}\nBot: {item['Bot']}" for item in context]
        )
        response = ""
        for ans in chain.stream({
            "context": formatted_context,
            "question": user_input
        }):
            print(ans, end="")
            response += ans

        print()

        if response.strip().startswith("text:"):
            yaml_file = save_yaml_response(response)
            read_and_write_code(yaml_file)

        context.append({"User": user_input, "Bot": response})

        if len(context) > MAX_CONTEXT_LENGTH:
            context = context[-MAX_CONTEXT_LENGTH:]

handle_conv()
