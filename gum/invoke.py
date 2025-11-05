from typing import List, Dict, Any, AnyStr
from openai import AsyncOpenAI
from datetime import datetime, timezone, timedelta


UTCm8 = timezone(timedelta(hours=-8))

async def invoke(
    client: AsyncOpenAI, 
    model: str, 
    messages: List[Dict[str, Any]], 
    response_format: dict,
    debug_tag: str = "",
    **kwargs,  # generalized variable to accept any further args
    ): 
    now = datetime.now(UTCm8) 
    print(now, debug_tag, "request sent")
    response = await client.chat.completions.create(
        model=model,
        messages = messages,
        response_format = response_format,   
        
        max_tokens = 10000 # TODO: adjust this too? 
        frequency_penalty = 1, # TODO: adjust this 
        temperature = 0.01,
        **kwargs, 
    )
    print(now, debug_tag, "response received")
    
    return response
    
    
    
    
# Based on     
# rsp = await self.client.chat.completions.create(
#     model=self.model_name,
#     messages=[{"role": "user", "content": content}],
#     response_format={"type": "text"},
# )
# return rsp.choices[0].message.content
