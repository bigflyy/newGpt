import os 
import time
import asyncio
from typing import List
import subprocess # run services bat. нужно будет убрать и все в докер засунуть
import traceback

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
COLLECTION_NAME = "newsgpt"


# TODO: CURRENT
# api 
# Preserve database 
# whisper (via mic at least ) and with text extension   
# Читать


# Н
# md 

# Д
# банк достовернных вопросов ответов 
# что-то из пособия  

# TODO: (FUTURE) 
# - time logger for embedding and retriever 
# - whisper end point 
# - tool call rag 
# - multimodal model (images)
# - chunker with llm 
# - graph 
# - tools calling (to vdb)

# (DONE)
# - metadata about the headers in each chunk, so it includes sources
# - number chunks based on most similiar 
# - raw chunk

# INFO
# conda activate suchgpt 
# run qdrant docker 
# run whisper_server
# запускать из бэкэнда 

# pkill whisper-server
# docker stop $(docker ps -q --filter ancestor=qdrant/qdrant)

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
    def __init__(self, collection_name = COLLECTION_NAME):
        self.collection_name = collection_name
        self.qdrant_host = "localhost"
        self.qdrant_port = 6333
        self.qdrant_base = f"http://{self.qdrant_host}:{self.qdrant_port}"
        self.ollama_base = "http://localhost:11434"
        self.whisper_base = "http://localhost:8080"
        self.qdrant_process = None
        self.whisper_process = None
        self.__run_services()

        self.embeddings = OllamaEmbeddings(
            # model="qwen3-embedding:8b", num_ctx=4096,
            model="qwen3-embedding:0.6b", num_ctx=2048,
            base_url=self.ollama_base,
            keep_alive=-1,
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
            model="qwen3:8b",  
            # model="gpt-oss:20b", num_ctx=65536, reasoning='low',
            base_url=self.ollama_base,
            stream=True,
            callbacks=[TimeLogger()],
            keep_alive=-1,
            # токенов
            
        )
        self.retriever = None
        self.prompt = ChatPromptTemplate.from_template(
            """ You are a helpful assistant. 
                Answer the question using ONLY the provided context.
                If relevant informating is not provided - say so.
                Answer without paraphrasing but in a readable format.
                List the sources of all chunks YOU USED in your answer. Example: "Источники: \n Документ(ы): .... \n Лекция(и): .... \n Глава(ы: .... \n Подглава(ы): .... \n"
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
    def assign_header_metadata(self, chunks):
        """
        Присваиваем метаданные к чанкам (Лекция, глава, подглава)
        """
        current_h1 = None
        current_h2 = None
        current_h3 = None
        result = []
        
        for i, chunk in enumerate(chunks):
            lines = chunk.split('\n')
            
            # Initialize LISTS to store ALL headers in this chunk
            chunk_h1s = []
            chunk_h2s = []
            chunk_h3s = []
            
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                    
                if stripped.startswith('#'):
                    hash_count = 0
                    while hash_count < len(stripped) and stripped[hash_count] == '#':
                        hash_count += 1
                    
                    if hash_count < len(stripped) and stripped[hash_count] == ' ':
                        header_text = stripped[hash_count+1:].strip()
                        
                        if hash_count == 1:
                            current_h1 = header_text
                            current_h2 = None
                            current_h3 = None
                            chunk_h1s.append(header_text)  # ADD to list
                        elif hash_count == 2:
                            current_h2 = header_text
                            current_h3 = None
                            chunk_h2s.append(header_text)  # ADD to list
                        elif hash_count == 3:
                            current_h3 = header_text
                            chunk_h3s.append(header_text)  # ADD to list
            
            # CRITICAL CHANGE: Store LISTS in metadata instead of single values
            metadata = {
                'id': i,
                'header1_list': chunk_h1s or [current_h1] if current_h1 else [],
                'header2_list': chunk_h2s or [current_h2] if current_h2 else [],
                'header3_list': chunk_h3s or [current_h3] if current_h3 else [],
            }
            
            result.append({
                'text': chunk,
                'metadata': metadata
            })
        
        return result
    def _extract_chunks_from_md(self, file_path):
        """
        Extract chunks from a single Markdown file with multi-header support.
        Prepends structured source metadata to each chunk's text.
        
        Returns list of LangChain Documents with:
        - page_content: Source block + original text
        - metadata: Lists of all headers in chunk + filename
        """
        filename = os.path.basename(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Split by chunk delimiter (robust against whitespace variations)
        parts = re.split(r'^--\s*Чанк\s*--$', content, flags=re.MULTILINE)
        raw_chunks = [part.strip() for part in parts[1:] if part.strip()]

        # Get chunks with header lists in metadata
        enriched_chunks = self.assign_header_metadata(raw_chunks)

        documents = []
        for item in enriched_chunks:
            metadata = item['metadata']
            chunk_text = item['text']
            
            # Get ALL headers from metadata lists (handle empty cases)
            lectures = metadata.get('header1_list', [])
            chapters = metadata.get('header2_list', [])
            subchapters = metadata.get('header3_list', [])
            
            # Format for source block (comma-separated or fallback)
            chunk_id = metadata.get('id')
            lecture_str = ", ".join(lectures) if lectures else "Не указана"
            chapter_str = ", ".join(chapters) if chapters else "Не указана"
            subchapter_str = ", ".join(subchapters) if subchapters else "Не указана"
            
            # Build Russian source block
            source_block = (
                "Источник\n"
                f"Документ: {filename}\n"
                f"Лекция: {lecture_str}\n"
                f"Глава: {chapter_str}\n"
                f"Подглава: {subchapter_str}"
            )
            
            # Combine source block with original content
            full_text = f"{source_block}\n\n{chunk_text}"
            
            # Preserve header lists in metadata for filtering
            doc_metadata = {
                'filename': filename,
                'header1_list': lectures,  # List of all H1s in chunk
                'header2_list': chapters,  # List of all H2s in chunk
                'header3_list': subchapters,  # List of all H3s in chunk
                'raw_chunktext': chunk_text # сырой текст чанка
            }
            documents.append(Document(
                page_content=full_text,
                metadata=doc_metadata
            ))
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
        
        # if not self.is_whisper_running():
        #     self.whisper_process = subprocess.Popen(['run_whisper.bat'], shell=True)
        #     print("Waiting for whisper-server...")
        #     for i in range(max_retries):
        #         if self.is_whisper_running():  # ✅ Use your helper method
        #             print("whisper-server ready")
        #             break
        #         time.sleep(1)
        #     else:
        #         raise Exception("Whisper failed to start")
        
        print("Both services are ready!")
    def clear(self):
        """
        Clean up all resources and reset the object to initial state.
        This includes:
        1. Stopping background services (Qdrant, Whisper)
        """
        print("Clearing all resources...")
        try:
            # 1. Stop Whisper service if running
            if hasattr(self, 'whisper_process') and self.whisper_process:
                print("Stopping Whisper service...")
                if self.whisper_process.poll() is None:  # Process is still running
                    self.whisper_process.terminate()
                    try:
                        self.whisper_process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self.whisper_process.kill()
                        print("⚠️  Whisper process had to be force-killed")
                self.whisper_process = None
                print("✅ Whisper service stopped")
        
            # 2. Stop Qdrant service if running  
            if hasattr(self, 'qdrant_process') and self.qdrant_process:
                print("Stopping Qdrant service...")
                if self.qdrant_process.poll() is None:  # Process is still running
                    self.qdrant_process.terminate()
                    try:
                        self.qdrant_process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        self.qdrant_process.kill()
                        print("Qdrant process had to be force-killed")
                self.qdrant_process = None
                print("Qdrant service stopped")
        
            # 3. Close Qdrant client connection
            if hasattr(self, 'qdrant_client') and self.qdrant_client:
                try:
                    print("Closing Qdrant client connection...")
                    # Qdrant client doesn't have explicit close method, but we can delete it
                    del self.qdrant_client
                    self.qdrant_client = None
                    print("Qdrant connection closed")
                except Exception as e:
                    print(f"Error closing Qdrant connection: {e}")
            print("Clear operation completed!")
            
        except Exception as e:
            print(f"❌ Error during clear operation: {e}")
            traceback.print_exc()
            return False
        
        return True
        
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
    def create_or_load_collection(self, md_dir=None):
        """
        Create collection from Markdown files OR load existing collection if already exists
        :param md_dir: Directory containing .md files (required only if creating new collection)
        """
        # Check if collection already exists
        collections = self.qdrant_client.get_collections().collections
        collection_names = [col.name for col in collections]
        
        if self.collection_name in collection_names:
            print(f"Collection '{self.collection_name}' already exists. Loading existing data...")
            # Load existing collection without recreating
            self.vector_store = QdrantVectorStore(
                client=self.qdrant_client,
                collection_name=self.collection_name,
                embedding=self.embeddings,
            )
            self._init_chain()
            return self.vector_store
        else:
            # Only create new collection if md_dir is provided
            if not md_dir:
                raise ValueError(f"Collection '{self.collection_name}' doesn't exist and no md_dir provided for creation")
            
            print(f"Creating new collection '{self.collection_name}'...")
            self.documents.clear()
            
            # Process markdown files as before
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
            
            # Create collection 
            self.vector_store = QdrantVectorStore.from_documents(
                documents=self.documents,
                embedding=self.embeddings,
                collection_name=self.collection_name,
                url=f"http://{self.qdrant_host}:{self.qdrant_port}",
            )
            self._init_chain()
            return self.vector_store
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
    def _serialize_document(self, doc):
        """Convert LangChain Document to JSON-serializable dictionary"""
        return {
            "page_content": doc.page_content,
            "metadata": {
                key: str(value) if isinstance(value, (set, tuple)) else value
                for key, value in doc.metadata.items()
            }
        }
    
    async def ask_with_context(self, query: str):
        """
        Единая точка входа: СНАЧАЛА возвращает контекст, ПОТОМ стримит ответ

        Возвращает генератор, который:
        1. Сначала выдает контекстные документы
        2. Затем стримит токены ответа по одному
        3. В конце выдает полный результат с контекстом и ответом
        """
        # Шаг 1: Сначала получаем КОНТЕКСТНЫЕ ДОКУМЕНТЫ     
        context_docs = await self.retriever.ainvoke(query)  
        
        # Inject rank into the content string
        for i, doc in enumerate(context_docs):
            rank = i + 1  # 1-based ranking
            # doc.page_content = f"[Номер чанка по релевантности: {rank}]\n{doc.page_content}"
            doc.metadata['relevance_rank'] = i + 1

        serializable_context = [self._serialize_document(doc) for doc in context_docs]

        # Шаг 2: СРАЗУ ВЫДАЕМ КОНТЕКСТ через yield
        yield {
            "type": "context",
            "content": serializable_context,  # Теперь это реальные Document объекты
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
            # Говорят что когда модель иницилизирует streaming то она дает пустые токены
            if not token or token.strip() == "":
                continue
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
            "context": serializable_context,  # Оригинальные документы
            "complete": True
        }   
    def transcribe_audio(self, file_path):
        """
        Convert audio to WAV format and send to whisper.cpp server
        """

        
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
    

async def main():
    gpt = NewSGPT()
    gpt.create_or_load_collection("backend/final_md")
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
                    preview = doc["page_content"]  # FIXED HERE
                    print(f"\n📄 Chunk {i}/{len(context_docs)}")
                    print(f"📝 Content: {preview}")
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


