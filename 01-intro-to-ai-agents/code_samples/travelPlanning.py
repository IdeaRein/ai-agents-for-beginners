import os
import random
import asyncio
from typing import Annotated
from dotenv import load_dotenv
from openai import AsyncOpenAI
from semantic_kernel.agents import ChatCompletionAgent, ChatHistoryAgentThread
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.functions import kernel_function

# プラグイン定義
class DestinationsPlugin:
    def __init__(self):
        self.destinations = [
            "Barcelona, Spain", "Paris, France", "Berlin, Germany",
            "Tokyo, Japan", "Sydney, Australia", "New York, USA",
            "Cairo, Egypt", "Cape Town, South Africa",
            "Rio de Janeiro, Brazil", "Bali, Indonesia"
        ]
        self.last_destination = None

    @kernel_function(description="ランダムな旅行先を提供")
    def get_random_destination(self) -> Annotated[str, "ランダムな旅行先を返す"]:
        available_destinations = self.destinations.copy()
        if self.last_destination and len(available_destinations) > 1:
            available_destinations.remove(self.last_destination)
        destination = random.choice(available_destinations)
        self.last_destination = destination
        return destination

# 環境変数ロード
load_dotenv()

# OpenAIクライアント作成
client = AsyncOpenAI(
    api_key=os.environ.get("GITHUB_TOKEN"),
    base_url="https://models.inference.ai.azure.com/"
)

# AIサービス作成
chat_completion_service = OpenAIChatCompletion(
    ai_model_id="gpt-4o-mini",
    async_client=client,
)

# エージェント作成
agent = ChatCompletionAgent(
    service=chat_completion_service,
    plugins=[DestinationsPlugin()],
    name="TravelAgent",
    instructions="あなたは顧客のためにランダムな旅行プランを提案する親切なAIです。提案内容はすべて日本語で行ってください。",
)

# メイン処理
async def main():
    thread: ChatHistoryAgentThread | None = None
    user_inputs = ["Plan me a day trip."]

    for user_input in user_inputs:
        print(f"# User: {user_input}\n")
        first_chunk = True
        async for response in agent.invoke_stream(messages=user_input, thread=thread):
            if first_chunk:
                print(f"# {response.name}: ", end="", flush=True)
                first_chunk = False
            print(f"{response}", end="", flush=True)
            thread = response.thread
        print()
    await thread.delete() if thread else None

if __name__ == "__main__":
    asyncio.run(main())
