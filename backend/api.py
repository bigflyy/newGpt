from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse

from typing import List, Literal, Optional, Union, Any
from rag import NewSGPT
import json
from pydantic import BaseModel, Field, ValidationError
import requests

app_state = {
    "rag_gpt": {}
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Перед запуском сервера
    rag_gpt = NewSGPT()
    print("Startup complete: rag_gpt initialized")
    app_state["rag_gpt"] = rag_gpt
    
    yield  # FastAPI принимает запросы 

    # При закрытии сервера
    # Закрываем наш rag
    app_state["rag_gpt"].clear()
    print("Shutdown complete: resources cleaned up")

app = FastAPI(lifespan=lifespan)

"""
sync function inside async - block entire server, event loop  V
sync function inside sync (just def), - fastapi automatically run in an external threadpool (пул потоков) that is then awaited (so does block the event loop) - V

so making sync calls inside of async is the worst type of idea? 
you either have it sync, or you have it async

@app.post("/embed")
async def embed_request(batch : EmbedBatch):
    channel = app_state["rabbitmq"]["channel"]
    # STOPS ENTIRE SERVER ANY SYNC OPERATION, stops every async path operation
    time.sleep(10)
    channel.basic_publish(
        exchange="",
        routing_key="embedding_queue",
        body=json.dumps(batch.model_dump()),
    )
    return {"status": "queued", "job_id": batch.job_id, "batch_index": batch.batch_index}
"""
# Define the request model with validation
class AskRequest(BaseModel):
    question: str = Field(..., description="The question text")
    # model_type: Literal["ragGpt", "baseGpt"] = Field(
    #     default="ragGpt", 
    #     description="Type of model to use"
    # )
    # model_names: Union[List[str], str, None] = Field(
    #     default=None,
    #     description="Model names to use, can be string or list"
    # )
    # use_graph: bool = Field(
    #     default=True,
    #     description="Whether to use graph"
    # )
    # debug: bool = Field(
    #     default=False,
    #     description="Enable debug mode"
    # )
    
@app.post("/ask")
async def ask_question(request: dict):
    # Get answer generator
    try:
        answer_gen = app_state["rag_gpt"].ask_with_context(request["question"])
    except KeyError:
        raise HTTPException(400, "Missing 'question' field")
    except Exception as e:
        raise HTTPException(500, str(e))
    
    # Stream response
    async def generate():
        try:
            async for chunk in answer_gen:
                yield json.dumps(chunk) + "\n"
        except Exception as e:
            yield json.dumps({"error": str(e)}) + "\n"
    return StreamingResponse(generate(), media_type="application/json")

@app.post("/uploads")
async def upload_files(files: List[UploadFile] = File(...)):
    print(f"\nReceived {len(files)} file(s):")
    for i, file in enumerate(files, 1):
        print(f"{i}. Filename: {file.filename}")
        print(f"   Content type: {file.content_type}")
        
        # Read file contents to get size
        contents = await file.read()
        file_size = len(contents)
        print(f"   File size: {file_size} bytes")
        
        # Reset file pointer to beginning for potential further processing
        await file.seek(0)
    
    return {"message": f"Successfully received {len(files)} files"}

@app.get("/getLLMList")
def get_llms():
    # actually we can choose... in theory 
    return {"llms": "no_choice =) yet"}
@app.get("/health")
def health_check():
    return {"status":"alive"}