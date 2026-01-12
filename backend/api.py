from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse

from typing import List, Literal, Optional, Union, Any
from rag import NewSGPT
import json
from pydantic import BaseModel, Field, ValidationError
import requests

# RUN WITH python -m uvicorn api:app --reload --port 8000

app_state = {
    "rag_gpt": {}
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Перед запуском сервера
    rag_gpt = NewSGPT()
    print("Startup complete: rag_gpt initialized")
    app_state["rag_gpt"] = rag_gpt
    rag_gpt.create_or_load_collection()
    
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
"""
# Define the request model with validation

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
    """
    Ask a question and receive a streaming RAG response with context and answer tokens.
    
    This endpoint processes a question using Retrieval-Augmented Generation (RAG) and returns
    a streaming response with three distinct phases: context retrieval, token-by-token answer generation,
    and a final completion chunk with the full answer.
    
    **Request Format:**
    ```json
    {
        "question": "Your question text here"
    }
    ```
    
    **Streaming Response Format:**
    The response is a stream of newline-delimited JSON objects (NDJSON) with three possible types:
    
    1. **Context Chunk** (sent first, once):
       ```json
       {
           "type": "context",
           "content": [
               {
                   "page_content": "Full text content of retrieved document chunk",
                   "metadata": {
                       "filename": "source document name",
                       "header1_list": ["Top-level headers in chunk"],
                       "header2_list": ["Second-level headers in chunk"],
                       "header3_list": ["Third-level headers in chunk"],
                       "raw_chunktext": "Original chunk text without source metadata",
                       "relevance_rank": 1
                   }
               }
           ],
           "complete": false
       }
       ```
    
    2. **Token Chunks** (sent multiple times during generation):
       ```json
       {
           "type": "token",
           "content": "Next token/word fragment of the answer",
           "complete": false
       }
       ```
    
    3. **Completion Chunk** (sent last, once):
       ```json
       {
           "type": "complete",
           "content": "Full generated answer text",
           "context": [  // Same format as context chunk content
               {
                   "page_content": "...",
                   "metadata": {...}
               }
           ],
           "complete": true
       }
       ```
    
    **Error Handling:**
    - 400 Bad Request: Missing 'question' field in request body
    - 500 Internal Server Error: Any exception during RAG processing or LLM generation
    
    **Response Media Type:** `application/json` (streaming NDJSON format)
    """
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
                yield json.dumps(chunk, ensure_ascii=False) + "\n"
        except Exception as e:
            yield json.dumps({"error": str(e)}, ensure_ascii=False) + "\n"
    return StreamingResponse(generate(), media_type="application/json")

# @app.post("/uploads")
# async def upload_files(files: List[UploadFile] = File(...)):
#     print(f"\nReceived {len(files)} file(s):")
#     for i, file in enumerate(files, 1):
#         print(f"{i}. Filename: {file.filename}")
#         print(f"   Content type: {file.content_type}")
        
#         # Read file contents to get size
#         contents = await file.read()
#         file_size = len(contents)
#         print(f"   File size: {file_size} bytes")
        
#         # Reset file pointer to beginning for potential further processing
#         await file.seek(0)
    
#     return {"message": f"Successfully received {len(files)} files"}

@app.get("/getLLMList")
def get_llms():
    # actually we can choose... in theory 
    return {"llms": "no_choice =) yet"}
@app.get("/health")
def health_check():
    return {"status":"alive"}