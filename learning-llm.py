from typing import Any, List, Optional
from langchain.embeddings import LlamaCppEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.llms.llamacpp import LlamaCpp
from langchain_core.language_models import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from .sentence_type_classifier import SentenceTypeClassifier
import os
from enum import Enum
import copy

class LocalModels(Enum):
    MANTICORE = "manticore-f16_q4_0.bin"
    LLAMA_13B = "llama-2-13b-q4_0.bin"
    LLAMA_7B = "llama-2-7b-q4_0.bin"

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

class ChatAny(LLM):
    llm: LlamaCpp
    db: Chroma
    sentence_classifier:SentenceTypeClassifier

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        
        print(f'PROMPT SUBMITTED TO MODEL\n {prompt}')
        #result = self.llm(prompt, temperature  = 0.7,max_tokens=5000)['choices'][0]['text']
        result = []
        for text in self.llm.stream(prompt, stop=["Q:"]):
            result.append(text)
            print(text, end='')
        return ' '.join(text)
    
    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "llama-2"

class Chatbot:
    _model_location:str = None
    _model_parameters:dict = None
    _db:Chroma = None
    _chain:BaseRetrievalQA = None

    def __init__(self, model_location:str, model_parameters:dict):
        self._model_location = model_location
        self._model_parameters = model_parameters

    @property
    def chain(self):
        retriever = self.db.as_retriever(search_kwargs={'kuname': 5})

        llm = ChatAny(
            llm=LlamaCpp(model_path=self._model_location, **self._model_parameters),
            db=self.db,
            sentence_classifier=SentenceTypeClassifier('/Users/davidmorton/Downloads/bert_sequence_classifier_question_statement_en_3/', '.token'))

        chain = RetrievalQA.from_chain_type(
            llm=llm, 
            chain_type="stuff", 
            retriever=retriever,
            verbose=True,
        )

        return chain

    @property
    def db(self):
        if self._db is None:
            parameters = copy.deepcopy(self._model_parameters)

            disallowed_tokens = ['max_tokens', 'temperature']
            for k in disallowed_tokens:
                if k in parameters:
                    del parameters[k]

            llama = LlamaCppEmbeddings(model_path=self._model_location, **parameters)   
            
            basename = os.path.basename(llama.model_path).split('.')[0]

            chroma_path = f'./chromadb_learning/{basename}'

            self._db = Chroma(persist_directory=chroma_path, embedding_function=llama)

        return self._db
    
    def query(self, prompt:str):
        return self.chain.run(prompt)

m = LocalModels.LLAMA_13B.value

chatbot:Chatbot = Chatbot(os.path.join('/Users/davidmorton/source/models/', m), parameter_sets[m])

result = chatbot.query('Who is Abraham Lincoln?')

print(result)