from openai import OpenAI

client = OpenAI()
async def generate_pair(system_prompt, user_prompt):
    messages = [
        {"role": "assistant", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    response = client.chat.completions.create(
        model = "gpt-4o",
        messages = messages, # type: ignore
        temperature = 0.7
    )
    x = response.choices
