from os import getenv

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from docassist.sampling.usual import get_sampler

def main():
    sampler = get_sampler()

    raw_model = OpenAIChatModel(
        'openai/gpt-oss-120b',
        provider=OpenRouterProvider(api_key=getenv("OPENAI_API_KEY")),
    )

    sampled_model = sampler.over_model(raw_model)
    agent = Agent(sampled_model)
    resp1 = agent.run_sync("Hi there, cowboy")
    print(resp1) # AgentRunResult(output='Howdy, partner! 🤠 How can I be of service on this fine day?')
    resp2 = agent.run_sync("Hi there, cowboy")
    print(resp2) # AgentRunResult(output='Howdy, partner! 🤠 How can I be of service on this fine day?')
