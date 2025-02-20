from flask import Flask, render_template, request, jsonify, session
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import AssistantMessage, SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Required for session management

# Azure AI Model Inference API settings
ENDPOINT = "https://models.inference.ai.azure.com"
MODEL_NAME = "Llama-3.3-70B-Instruct"
AZURE_TOKEN = os.environ['GITHUB_TOKEN']
if not AZURE_TOKEN:
    raise ValueError("GITHUB_TOKEN environment variable not set")
# Set this in your environment

# Initialize Azure client
client = ChatCompletionsClient(
    endpoint=ENDPOINT,
    credential=AzureKeyCredential(AZURE_TOKEN),
)

# Function to get response from Azure AI Model Inference API using SDK
def get_azure_response(user_message):
    # Ensure conversation history exists
    if 'conversation' not in session:
        session['conversation'] = []

    # System message to define the assistant's role
    system_message = SystemMessage(content="You are a helpful assistant.")

    # Convert session history to Azure message format
    history = []
    for msg in session['conversation']:
        if msg['role'] == 'user':
            history.append(UserMessage(content=msg['content']))
        elif msg['role'] == 'assistant':
            history.append(AssistantMessage(content=msg['content']))

    # Add current user message
    user_message_obj = UserMessage(content=user_message)

    # Combine messages
    messages = list(system_message  + history + user_message_obj)
    print(messages)

    try:
        response = client.complete(messages=messages, model=MODEL_NAME, temperature=0.7)
        assistant_message = response.choices[0].message.content

        # Update conversation history
        session['conversation'].append({"role": "user", "content": user_message})
        session['conversation'].append({"role": "assistant", "content": assistant_message})
        return assistant_message
    except Exception as e:
        return f"Error calling Azure API: {str(e)}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['message']
    response = get_azure_response(user_message)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)