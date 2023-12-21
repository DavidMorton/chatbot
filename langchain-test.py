from typing import Any, List, Optional
from llama_cpp import Llama #import for GGML models
from llama_cpp.llama_chat_format import LlamaChatCompletionHandler
from langchain.embeddings import LlamaCppEmbeddings
from langchain.text_splitter import CharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
from langchain.llms.llamacpp import LlamaCpp
from langchain_core.language_models import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain.document_loaders import PyPDFLoader
import nltk
from pydantic import PrivateAttr
import os, shutil
from enum import Enum
import copy

class LocalModels(Enum):
    MANTICORE = "manticore-f16_q4_0.bin"
    LLAMA_13B = "llama-2-13b-q4_0.bin"
    LLAMA_7B = "llama-2-7b-q4_0.bin"

selected_model = LocalModels.LLAMA_13B

parameter_sets = {
    LocalModels.LLAMA_7B.value: {
        'temperature': 0.7,
        'max_tokens': 5000,
        'n_batch': 512,
        'n_ctx': 12000,
        'n_gpu_layers': 1
    },
    LocalModels.LLAMA_13B.value: {
        'temperature': 0.7,
        'max_tokens': 5000,
        'n_batch': 512,
        'n_ctx': 4000,
        'n_gpu_layers': 1
    }
}

models_root = '/Users/davidmorton/source/models/'

model_location = os.path.join(models_root, selected_model.value)

class ChatAny(LLM):
    llm: LlamaCpp

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        print(f'PROMPT SUBMITTED TO MODEL\n {prompt}')
        #result = self.llm(prompt, temperature  = 0.7,max_tokens=5000)['choices'][0]['text']
        for text in self.llm.stream(prompt, stop=["Q:"]):
            print(text, end='')
        return ''
    
    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "llama-2"

def get_chroma_db(model:LocalModels, recreate_chroma=False):

    parameters = copy.deepcopy(parameter_sets[model.value])
    disallowed_tokens = ['max_tokens', 'temperature']
    for k in disallowed_tokens:
        if k in parameters:
            del parameters[k]

    llama = LlamaCppEmbeddings(model_path=model_location, **parameters)   
    
    basename = os.path.basename(llama.model_path).split('.')[0]

    chroma_path = f'./chromadb/{basename}'

    raw_documents = TextLoader('cv.txt').load()

    text_splitter = SentenceTransformersTokenTextSplitter(chunk_size=512, chunk_overlap=0)

    documents = text_splitter.split_documents(raw_documents)

    recreate_chroma = recreate_chroma or not os.path.exists(chroma_path)

    if recreate_chroma:
        shutil.rmtree(chroma_path, ignore_errors=True)
        db = Chroma(embedding_function=llama, persist_directory=chroma_path)

        for document in documents:
            db.add_documents([document])

        for file in os.listdir('./cover-letters'):
            print(f'Adding {file}...')
            filepath = os.path.join('./cover-letters', file)
            loader = PyPDFLoader(filepath)
            cover_letter_document = loader.load()
            documents = text_splitter.split_documents(cover_letter_document)
            for document in documents:
                db.add_documents([document])
    else:
        db = Chroma(persist_directory=chroma_path, embedding_function=llama)

    return db

db = get_chroma_db(selected_model, recreate_chroma=False)

retriever = db.as_retriever(search_kwargs={'kuname': 5})

llm = ChatAny(llm=LlamaCpp(model_path=model_location, **parameter_sets[selected_model.value]))

qa_stuff = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever,
    verbose=True,
)

query = None

while query not in ['quit','exit']:
    if query not in [None, '']:
        print(qa_stuff.run(query))
    query = input('Prompt:')