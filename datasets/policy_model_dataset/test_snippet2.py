from openai import OpenAI

client = OpenAI()
async def generate_pair(system_prompt: str, user_prompt: str):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    response = client.chat.completions.create(
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        model = "gpt-4o",
        temperature = 0.7
    )
    x = response.choices
