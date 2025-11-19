# Auth: Andy Phu
# Introduced to centralize communications to VLLM server

from typing import List, Dict, Any, AnyStr
from openai import AsyncOpenAI
from datetime import datetime, timezone, timedelta
import logging

UTCm8 = timezone(timedelta(hours=0))

async def invoke(
    client: AsyncOpenAI, 
    model: str, 
    messages: List[Dict[str, Any]], 
    response_format: dict,
    debug_tag: str = "",
    **kwargs,  # generalized variable to accept any further args
    ): 
    
    logger = logging.getLogger("Invoke")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s - [VIA-INVOKE] - %(message)s"))
        logger.addHandler(h)
        
    now = datetime.now(UTCm8) 
    # timestamp = f"{now.strftime('%H:%M:%S')}"
    logger.info(f"{debug_tag} request sent")
    response = await client.chat.completions.create(
        model=model,
        messages = messages,
        response_format = response_format,   
        
        max_tokens = 10000, # TODO: adjust this too? 
        frequency_penalty = 1, # TODO: adjust this 
        temperature = 0.01,
        **kwargs, 
    )
    logger.info(f"{debug_tag} response received")
    
    
    
    return response
    
    
    
    
# Based on     
# rsp = await self.client.chat.completions.create(
#     model=self.model_name,
#     messages=[{"role": "user", "content": content}],
#     response_format={"type": "text"},
# )
# return rsp.choices[0].message.content
