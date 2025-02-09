import os
from mistralai import Mistral

api_key = os.getenv("MISTRAL_API_KEY")
client = Mistral(api_key=api_key)

model = "codestral-latest"
prompt = "def fibonacci(n: int):"
suffix = "n = int(input('Enter a number: '))\nprint(fibonacci(n))"

response = client.fim.complete(
    model=model,
    prompt=prompt,
    suffix=suffix,
    temperature=0,
    top_p=1,
)

print(
    f"""
{prompt}
{response.choices[0].message.content}
{suffix}
"""
)