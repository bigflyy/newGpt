import os 
import time
import asyncio
from typing import List
import subprocess # run services bat. нужно будет убрать и все в докер засунуть

# langchain and rag stuff
import re

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.runnables import RunnableLambda, RunnableParallel

# WHISPER STUFF
from pydub import AudioSegment
import requests
import numpy as np
import soundfile as sf
import io


# -----------------------
# Configuration
# -----------------------

OLLAMA_BASE = "http://localhost:11434"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "my_docs"


# TODO:
# - time logger for embedding and retriever 
# - metadata about the headers in each chunk, so it includes sources
# - whisper end point 
# - tool call rag 

# INFO
# conda activate suchgpt 
# run qdrant docker 
# run whisper_server
# запускать из бэкэнда 

#pkill whisper-server
#docker stop $(docker ps -q --filter ancestor=qdrant/qdrant)

class TimeLogger(BaseCallbackHandler):
    """
    Логирование времени для каждого из этапов RAG'а
    """
    def __init__(self):
        super().__init__()
        self.start_times = {}
    
    def on_retriever_start(self, serialized, inputs, **kwargs):
        self.start_times['retriever'] = time.perf_counter()

    def on_retriever_end(self, outputs, **kwargs):
        elapsed = time.perf_counter() - self.start_times.get('retriever', 0)
        print(f"# Retriever time: {elapsed:.3f}s")

    def on_llm_start(self, serialized, inputs, **kwargs):
        self.start_times['llm'] = time.perf_counter()

    def on_llm_end(self, outputs, **kwargs):
        elapsed = time.perf_counter() - self.start_times.get('llm', 0)
        print(f"# LLM time: {elapsed:.3f}s")
    
    # NEW: Add embedding timing methods
    def on_embedding_start(self, serialized, inputs, **kwargs):
        self.start_times['embedding'] = time.perf_counter()

    def on_embedding_end(self, outputs, **kwargs):
        elapsed = time.perf_counter() - self.start_times.get('embedding', 0)
        print(f"# Embedding time: {elapsed:.3f}s")
    
# # Simple callback to print tokens live
# class StreamingStdOutCallback(BaseCallbackHandler):
#     def on_model_new_token(self, token: str, **kwargs):
#         print(token, end="", flush=True)


class NewSGPT:
    def __init__(self):
        self.collection_name = "newsgpt"
        self.qdrant_host = "localhost"
        self.qdrant_port = 6333
        self.qdrant_base = f"http://{self.qdrant_host}:{self.qdrant_port}"
        self.ollama_base = "http://localhost:11434"
        self.whisper_base = "http://localhost:8080"
        self.qdrant_process = None
        self.whisper_process = None
        self.__run_services()

        self.embeddings = OllamaEmbeddings(
            model="qwen3-embedding:8b",
            base_url=self.ollama_base,
            num_ctx=4096
        )
        self.qdrant_client = QdrantClient(
            host=self.qdrant_host,
            port=self.qdrant_port,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
        )
        self.documents: list[Document] = []
        # Векторная база данных 
        self.vector_store = None
        # LLM
        self.model_llm = ChatOllama(
            # model="deepseek-r1:8b",   
            model="gpt-oss:20b",
            base_url=OLLAMA_BASE,
            reasoning='low',
            stream=True,
            callbacks=[TimeLogger()],
            keep_alive=-1,
            num_ctx=65536, # токенов
            
        )
        self.retriever = None
        self.prompt = ChatPromptTemplate.from_template(
            """ You are a helpful assistant. 
                Answer the question using ONLY the provided context.
                If relevant informating is not provided - say so.
                Answer without paraphrasing but in a readable format.
                Write source.
                Answer in Russian only. 

                Context:
                {context}

                Question:
                {question}
            """
        )
        # LCEL RAG chain
        self.rag_chain = None

        self.context_chain = None
    def _extract_chunks_from_md(self, file_path) -> List[Document]:
        """
        Docstring for _extract_chunks_from_md
        
        :param file_path: path to md file
        
        returns Documents of chunks
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split the content by the delimiter
        # Using regex to handle possible whitespace around the delimiter
        parts = re.split(r'^--\s*Чанк\s*--$', content, flags=re.MULTILINE)

        # The first part is content before the first "-- Чанк --" (if any)
        # We skip it if it's just leading noise
        chunks = [part.strip() for part in parts[1:] if part.strip()]

        documents = []

        for idx, chunk in enumerate(chunks, 1):
            # Create document with comprehensive metadata
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": file_path,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "file_type": "markdown",
                    "chunk_delimiter": "-- Чанк --",
                }
            )
            documents.append(doc)
        return documents
    def _pdfs_to_txts(self, pdfs_folder, txts_folder):
        # Create output directory if needed
        os.makedirs(txts_folder, exist_ok=True)
        # Convert all PDFs in the pdfs folder
        for filename in os.listdir(pdfs_folder):
            if filename.lower().endswith('.pdf'):
                pdf_path = os.path.join(pdfs_folder, filename)
                txt_filename = os.path.splitext(filename)[0] + '.txt'
                txt_path = os.path.join(txts_folder, txt_filename)
                NewSGPT.pdf_to_txt(pdf_path, txt_path)
                print(f"Converted: {filename} -> {txt_filename}")
    
    ##### это все кринж это надо будет убрать и сделать docker compose но мне лень #### 
    def is_whisper_running(self):
        try:
            response = requests.get(f"{self.whisper_base}/health", timeout=1)
            return response.status_code == 200  # Check status code, not the response object
        except (requests.exceptions.RequestException, requests.exceptions.Timeout):
            return False  # Properly catch network errors
    def is_qdrant_running(self):
        try:
            response = requests.get(f"{self.qdrant_base}/healthz", timeout=1)
            return response.status_code == 200
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                requests.exceptions.RequestException):
            return False
    def __run_services(self):
        """
        Запускает whisper и qdrant
        """
        max_retries = 360
        
        if not self.is_qdrant_running():
            self.qdrant_process = subprocess.Popen(['run_qdrant.bat'], shell=True)
            print("Waiting for Qdrant...")
            for i in range(max_retries):
                if self.is_qdrant_running():  # ✅ Use your helper method
                    print("Qdrant ready")
                    break
                time.sleep(1)
            else:
                raise Exception("Qdrant failed to start")
        
        if not self.is_whisper_running():
            self.whisper_process = subprocess.Popen(['run_whisper.bat'], shell=True)
            print("Waiting for whisper-server...")
            for i in range(max_retries):
                if self.is_whisper_running():  # ✅ Use your helper method
                    print("whisper-server ready")
                    break
                time.sleep(1)
            else:
                raise Exception("Whisper failed to start")
        
        print("Both services are ready!")
        ######################################################################
    def _init_chain(self):
        """
        Создает цепь с поддержкой стриминга и возврата контекста
        """
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        
        # Цепочка для получения контекста
        self.context_chain = (
            {"question": RunnablePassthrough()}
            | RunnableLambda(lambda x: self.retriever.ainvoke(x["question"]))
        )
        
        # Цепочка для стриминга ответа
        self.answer_chain = (
            {
                "context": lambda x: "\n\n".join([doc.page_content for doc in x["context"]]),
                "question": lambda x: x["question"]
            }
            | self.prompt
            | self.model_llm  # stream=True уже установлено в инициализации
            | StrOutputParser()
        )
    #################################
    
    def create_collection_from_mds(self, md_dir):
        """
        Create Qdrant collection from Markdown files containing chunk-delimited content
        Creates self.vector_store
        
        :param md_dir: Directory containing .md files with chunks marked by "-- Чанк --"
        """
        self.documents.clear()  # Reset documents list
        
        for filename in os.listdir(md_dir):
            if not filename.endswith('.md'):
                continue
            file_path = os.path.join(md_dir, filename)
            try:
                file_documents = self._extract_chunks_from_md(file_path)
                self.documents.extend(file_documents)
                print(f"  Extracted {len(file_documents)} chunks from {filename}")

            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
        
        if not self.documents:
            raise ValueError("No valid chunks found in Markdown files")
        
        # Create Qdrant collection
        self.vector_store = QdrantVectorStore.from_documents(
            documents=self.documents,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            url=f"http://{self.qdrant_host}:{self.qdrant_port}",
            force_recreate=True,  # Recreate collection each time
        )
        self._init_chain()
        return self.vector_store
    async def answer(self, query : str):
        async for token in self.rag_chain.astream(query):
            yield token
    async def ask_with_context(self, query: str):
        """
        Единая точка входа: СНАЧАЛА возвращает контекст, ПОТОМ стримит ответ
        
        Возвращает генератор, который:
        1. Сначала выдает контекстные документы
        2. Затем стримит токены ответа по одному
        3. В конце выдает полный результат с контекстом и ответом
        """
        # Шаг 1: Сначала получаем КОНТЕКСТНЫЕ ДОКУМЕНТЫ (ОБЯЗАТЕЛЬНО await!)
        context_docs = await self.retriever.ainvoke(query)  #  КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: await здесь
        
        # Шаг 2: СРАЗУ ВЫДАЕМ КОНТЕКСТ через yield
        yield {
            "type": "context",
            "content": context_docs,  # Теперь это реальные Document объекты
            "complete": False
        }
        
        # Шаг 3: Формируем промпт с контекстом
        context_text = "\n\n".join([doc.page_content for doc in context_docs])
        messages = self.prompt.format_messages(
            context=context_text,
            question=query
        )
        
        # Шаг 4: Стримим ответ от LLM
        full_answer = ""
        async for chunk in self.model_llm.astream(messages):
            token = chunk.content if hasattr(chunk, 'content') else str(chunk)
            full_answer += token
            yield {
                "type": "token", 
                "content": token,
                "complete": False
            }
        
        # Шаг 5: В конце возвращаем полный результат
        yield {
            "type": "complete",
            "content": full_answer,
            "context": context_docs,  # Оригинальные документы
            "complete": True
        }
    def transcribe_audio(self, file_path):
        """
        Convert audio to WAV format and send to whisper.cpp server
        """
        from pydub import AudioSegment
        import io
        import requests
        
        # Convert to WAV format (required by whisper.cpp)
        audio = AudioSegment.from_file(file_path)
        audio = audio.set_frame_rate(16000).set_channels(1)  # Mono, 16kHz
        
        # Save to in-memory WAV buffer
        buffer = io.BytesIO()
        audio.export(buffer, format="wav")
        buffer.seek(0)
        
        # Send to whisper.cpp server - CORRECTED VERSION
        files = {'file': ('audio.wav', buffer, 'audio/wav')}  # Field name must be 'file'
        data = {
            'response_format': 'json',  # Required for JSON response
        }
        
        response = requests.post(
            f"{self.whisper_base}/inference", 
            files=files,
            data=data
        )
        
        if response.status_code == 200:
            return response.json()['text']
        else:
            raise Exception(f"Transcription failed: {response.text}")
    def clear():
        #TODO: close whisper, qdrant, offload models from vram
        pass
async def main():
    gpt = NewSGPT()
    gpt.create_collection_from_mds("final_md")
    
    while True:
        query = input("\nAsk a question (or 'quit' to exit): ").strip()
        if not query or query.lower() == "quit":
            break
        
        print("\n" + "="*80)
        print(f"🔍 Question: {query}")
        
        context_shown = False
        full_answer = ""
        
        # ВАЖНО: Используем async for для обработки генератора
        async for result in gpt.ask_with_context(query):
            if result["type"] == "context" and not context_shown:
                # Теперь result["content"] - это реальные Document объекты
                context_docs = result["content"]
                context_shown = True
                
                print("\n" + "-"*80)
                print(f"📚 Retrieved {len(context_docs)} context chunks:")  # ← Теперь len() работает!
                for i, doc in enumerate(context_docs, 1):
                    source = os.path.basename(doc.metadata.get('source', 'unknown'))
                    preview = doc.page_content + "..." if len(doc.page_content) > 150 else doc.page_content
                    print(f"\n📄 Chunk {i}/{len(context_docs)}")
                    print(f"📁 Source: {source}")
                    print(f"📝 Preview: {preview}")
                print("\n" + "-"*80)
                print("💡 Generating answer using this context...")
                print("-"*80)
                
            elif result["type"] == "token":
                if not context_shown:
                    # На случай, если контекст не был показан (защита)
                    context_shown = True
                    print("\n" + "-"*80)
                    print("💡 Generating answer...")
                    print("-"*80)
                print(result["content"], end="", flush=True)
                full_answer += result["content"]
                
            elif result["type"] == "complete":
                print()  # Новая строка после стриминга
                print("\n" + "="*80)
                print(f"✅ Complete! Generated {len(full_answer)} characters")
                print(f"📊 Used {len(result['context'])} context chunks")

if __name__ == "__main__":
    asyncio.run(main())


