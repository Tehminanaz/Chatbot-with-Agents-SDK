import os
import chainlit as cl
from dotenv import load_dotenv, find_dotenv
from agents import Agent, RunConfig, AsyncOpenAI, OpenAIChatCompletionsModel, Runner

load_dotenv(find_dotenv())

gemini_api_key = os.getenv("GEMINI_API_KEY")

provider = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider
)

run_config = RunConfig(
    model=model,
    model_provider=provider,
    tracing_disabled=True
)

prompt_optimizer_agent = Agent(
    name="Prompt Optimization Assistant",
    instructions="You analyze prompts and suggest improvements for better LLM responses."
)

@cl.on_chat_start
async def handle_chat_start():
    cl.user_session.set("history", [])
    await cl.Message(content="Hello! I am your Prompt Optimization Assistant. Send me a prompt, and I'll help improve it!").send()

@cl.on_message
async def handle_message(message: cl.Message):
    history = cl.user_session.get("history")
    
    history.append({"role": "user", "content": message.content})
    
    result = await Runner.run(prompt_optimizer_agent, input=history, run_config=run_config)
    
    history.append({"role": "assistant", "content": result.final_output})
    cl.user_session.set("history", history)
    
    await cl.Message(content=f"Suggested Improvement: {result.final_output}").send()