# Auth: Andy Phu
# Introduced to centralize communications to VLLM server

from typing import List, Dict, Any, AnyStr
from openai import AsyncOpenAI
from datetime import datetime, timezone, timedelta
import logging
from .data import save_to_file, copy_imgs
from pathlib import Path

UTCm0 = timezone(timedelta(hours=0))

async def invoke(
    client: AsyncOpenAI, 
    model: str, 
    messages: List[Dict[str, Any]], 
    response_format: dict,
    debug_tag: str = "",
    debug_img_paths: str = "", 
    debug_path: Path = "",
    **kwargs,  # generalized variable to accept any further args
    ): 
    
    logger = logging.getLogger("Invoke")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(h)
        
    now = datetime.now(UTCm0) 
    ts = now.strftime("%Y-%m-%d %H:%M:%S")
    request_fts = now.strftime("%Y%m%d_%H%M%S")
    
    subfolder_path = Path(debug_path) / f"{request_fts}-{debug_tag}"
    
    if debug_tag == "[Retro Transcription]" or debug_tag == "[Retro Summary]": #TODO: modify filter? 
        
        if(debug_img_paths): 
            fp = save_to_file(text=f"{messages}", subfolder=subfolder_path, filename=f"{request_fts}-{debug_tag}-SND.txt")
            copy_imgs(img_paths = debug_img_paths, subfolder=subfolder_path )
        else:
            fp = save_to_file(text=f"{messages}", subfolder=subfolder_path, filename=f"{request_fts}-{debug_tag}-SND.txt")
              
    logger.info(f"{ts} [VIA-INVOKE] {debug_tag} request sent, img_paths: {debug_img_paths}")
    
    response = await client.chat.completions.create(
        model=model,
        messages = messages,
        response_format = response_format,   
        
       #max_tokens = 10000, # TODO: adjust this too? 
        frequency_penalty = 0.01, # TODO: adjust this 
        temperature = 0.1,
        **kwargs, 
    )
    
    
    now = datetime.now(UTCm0) 
    response_fts = now.strftime("%Y%m%d_%H%M%S")
    ts = now.strftime("%Y-%m-%d %H:%M:%S")
    
    subfolder_path = Path(debug_path) / f"{request_fts}-{debug_tag}"
    
    
    if debug_tag == "[Retro Transcription]" or debug_tag == "[Retro Summary]":
        
        fp = save_to_file(text=str(response.choices[0].message.content), subfolder=subfolder_path, filename=f"{response_fts}-{debug_tag}-RCV.txt")
    
    logger.info(f"{ts} [VIA_INVOKE] {debug_tag} response received, stored at {fp}")
    
    
    return response
    
    
    
    
# Based on     
# rsp = await self.client.chat.completions.create(
#     model=self.model_name,
#     messages=[{"role": "user", "content": content}],
#     response_format={"type": "text"},
# )
# return rsp.choices[0].message.content
