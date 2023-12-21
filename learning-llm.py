from typing import Any, List, Optional
from langchain.embeddings import LlamaCppEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.chains.retrieval_qa.base import BaseRetrievalQA
from langchain.llms.llamacpp import LlamaCpp
from langchain_core.language_models import LLM
from langchain_core.callbacks import CallbackManagerForLLMRun
from sentence_type_classifier import SentenceTypeClassifier
from langchain.text_splitter import CharacterTextSplitter, SentenceTransformersTokenTextSplitter
from langchain.document_loaders import TextLoader
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
        'n_gpu_layers': 1,
        'repeat_penalty': 1.3
    }
}

class ChatAny(LLM):
    llm: LlamaCpp
    db: Chroma

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        #prompt = prompt.replace("Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.", "Context:")
        #prompt = prompt.replace("\nHelpful Answer:", "")
        #result = self.llm(prompt, temperature  = 0.7,max_tokens=5000)['choices'][0]['text']
        result = []
        for text in self.llm.stream(prompt):
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
    _classifier_path:str = None
    _token_file:str = None
    _classifier:SentenceTypeClassifier = None

    def __init__(self, model_location:str, model_parameters:dict, classifier_path:str, token_file:str):
        self._model_location = model_location
        self._model_parameters = model_parameters
        self._classifier_path = classifier_path
        self._token_file = token_file

    @property
    def classifier(self):
        if self._classifier is None:
            self._classifier = SentenceTypeClassifier(self._classifier_path, self._token_file)
        return self._classifier
    
    @property
    def chain(self):
        if self._chain is None:
            retriever = self.db.as_retriever(search_kwargs={'kuname': 5})

            llm = ChatAny(
                llm=LlamaCpp(model_path=self._model_location, top_k=2, **self._model_parameters),
                db=self.db,
                sentence_classifier=SentenceTypeClassifier('/Users/davidmorton/Downloads/bert_sequence_classifier_question_statement_en_3/', '.token'))

            self._chain = RetrievalQA.from_chain_type(
                llm=llm, 
                chain_type="stuff", 
                retriever=retriever,
                verbose=True,
            )

        return self._chain

    @property
    def db(self):
        if self._db is None:
            parameters = copy.deepcopy(self._model_parameters)
            allowed_tokens = list(LlamaCppEmbeddings.__fields__.keys())
            for k in self._model_parameters:
                if k not in allowed_tokens:
                    del parameters[k]

            llama = LlamaCppEmbeddings(model_path=self._model_location, **parameters)   
            
            basename = os.path.basename(llama.model_path).split('.')[0]

            chroma_path = f'./chromadb_learning/{basename}'

            self._db = Chroma(persist_directory=chroma_path, embedding_function=llama)

        return self._db
    
    def add_information(self, text):
        self.db.add_texts([text])
    
    def query(self, prompt:str):
        classification_result = self.classifier.classify(prompt)
        if classification_result == 1:
            return self.chain.run(prompt)
        
        self.add_information(prompt)
        print('Okay. I\'ll remember that.')
        return 'Okay. I\'ll remember that.'
    
    def start_chat(self):
        prompt = ''
        while prompt.lower() not in ['bye','exit','quit','goodbye']:
            if prompt.strip() != '':
                self.query(prompt)
            prompt = input('Prompt: ')

m = LocalModels.LLAMA_13B.value

chatbot:Chatbot = Chatbot(
    os.path.join('/Users/davidmorton/source/models/', m), 
    parameter_sets[m], 
    "shahrukhx01/question-vs-statement-classifier", 
    '.token')

chatbot.start_chat()