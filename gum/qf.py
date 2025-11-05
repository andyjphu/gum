from openai import AsyncOpenAI
from invoke import invoke
import asyncio

        # rsp = await invoke(
        #     model=self.model_name,
        #     messages=[{"role": "user", "content": content}],
        #     response_format={"type": "text"}, 
        #     debug_tag="[Manual]",
        #     client=self.client,
        # )
        
# gum  --user-name "andrew" --model "Qwen/Qwen2.5-VL-7B-Instruct"
        #
client = AsyncOpenAI(
    base_url="http://127.0.0.1:8000/v1",
    api_key="None"
)

async def main():
    rsp = await invoke(
        model="Qwen/Qwen2.5-VL-7B-Instruct", #
        messages=[{"role": "user", "content": "test"}],
        response_format={"type": "text"}, 
        debug_tag="[Test]",
        client=client,
    )
    
    print(rsp)    



asyncio.run(main())