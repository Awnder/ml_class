import openai

client = openai.OpenAI(api_key="")

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system", "content": "You are a Shakespearean actor.",
            "role": "user", "content": "What is love?",
        }
    ]
)

print(response.choices[0].message.content)