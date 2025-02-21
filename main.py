from flask import Flask, render_template, request, jsonify, session, Response
import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import AssistantMessage, SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError
import json

app = Flask(__name__)
app.secret_key = 'your-secret-key'  # Required for session management

# Azure AI Model Inference API settings
ENDPOINT = "https://models.inference.ai.azure.com"
MODEL_NAME = "Llama-3.3-70B-Instruct"
AZURE_TOKEN = os.environ.get('GITHUB_TOKEN')

# Check if token is set
if not AZURE_TOKEN:
    raise ValueError("AZURE_TOKEN environment variable is not set. Please set it with a valid API key.")

# Initialize Azure client
try:
    client = ChatCompletionsClient(
        endpoint=ENDPOINT,
        credential=AzureKeyCredential(AZURE_TOKEN),
    )
except Exception as e:
    raise ValueError(f"Failed to initialize Azure client: {str(e)}")

# Function to get streaming response from Azure AI Model Inference API using SDK
def get_azure_streaming_response(user_message):
    # Ensure conversation history exists
    if 'conversation' not in session:
        session['conversation'] = []

    # System message to define the assistant's role
    system_message = SystemMessage(content="You are a helpful assistant. Respond as Kim, saying 'Hi, I'm Kim - ask me anything, I'm happy to help analyze and discuss any webpages, images, or code you share with me.' for the first message if no prior conversation exists.")

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
    messages = [system_message] + history + [user_message_obj]

    try:
        # Stream response from Azure API
        response = client.complete(messages=messages, model=MODEL_NAME, temperature=0.7, stream=True)

        def event_stream():
            full_response = ""
            for chunk in response:
                if chunk.choices:
                    content = chunk.choices[0].delta.content or ""
                    full_response += content
                    yield f"data: {json.dumps({'content': content})}\n\n"
            # Update conversation history with full response
            session['conversation'].append({"role": "user", "content": user_message})
            session['conversation'].append({"role": "assistant", "content": full_response})
            yield "data: [DONE]\n\n"

        return Response(event_stream(), mimetype="text/event-stream")

    except HttpResponseError as e:
        return jsonify({"error": f"Azure API error: {e.status_code} - {e.message}"})
    except Exception as e:
        return jsonify({"error": f"Error calling Azure API: {str(e)}"})
        
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['GET'])
def chat():
    user_message = request.args.get('message')
    # Stream the response
    return get_azure_streaming_response(user_message)

if __name__ == '__main__':
    app.run(debug=True)