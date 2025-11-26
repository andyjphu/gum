import asyncio
from sqlalchemy import select
from gum.models import Proposition, Observation, init_db
import json

async def dump_db():
    engine, Session = await init_db("gum.db", ".cache/gum")
    
    async with Session() as session:
        # Get all propositions
        props_result = await session.execute(select(Proposition))
        props = props_result.scalars().all()
        
        # Get all observations
        obs_result = await session.execute(select(Observation))
        observations = obs_result.scalars().all()
        
        dump = {
            "propositions": [
                {
                    "id": p.id,
                    "text": p.text,
                    "reasoning": p.reasoning,
                    "confidence": p.confidence,
                    "decay": p.decay,
                    "version": p.version,
                    "updated_at": str(p.updated_at)
                }
                for p in props
            ],
            "observations": [
                {
                    "id": o.id,
                    "observer_name": o.observer_name,
                    "content": o.content,
                    "created_at": str(o.created_at)
                }
                for o in observations
            ]
        }
        
       # print()
        
        with open ("./data.json", "w") as f:
            f.write(json.dumps(dump, indent=2))

asyncio.run(dump_db())