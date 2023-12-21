# Everybody's doing it, so why shouldn't I?

Everybody is building an LLM, so why not me?

## My personal LLM experimentation Repo

This is a WIP, "learning" chatbot. The idea behind this is to combine BERT text classification methods with a small LLM model to build out a chatbot that can effectively work. 

The pathway here is as follows:

1. Prompt the user.
1. Determine if the prompt entered is a question or a statement.
    1. If it's a statement, then tokenize it and add it to the Chroma database.
    1. If it's a question, then query the vector database for applicable context, and ask the LLM to answer the question.

The hope is that, over time, the user will be able to effectively teach the Chatbot the data that it needs to know, through a more conversational approach, creating a "Jarvis-like" personal assistant. 

### Notes on my environment

I'm running on a Mac M2, which makes LLM-things... unique. First off, there's no direct access to the GPU, so the llama.cpp project has been helpful as a workaround. That said, loading memory into Metal is somewhat challenging. There's a 16GB hard limit in the GPU. This creates certain limitations.

* [LLaMA 2 13B](https://huggingface.co/TheBloke/Llama-2-13B-GGUF/tree/main) seems to run alright, but only when there's extremely limited context given (<~4000 tokens)
* [LLaMA 2 7B](https://huggingface.co/TheBloke/Llama-2-7B-GGUF/tree/main) allows around 12000 tokens max.

### Setup

Install miniconda. 

Run the following command:
```conda create -f ./environment.yml```

Run the learning-llm.py file. 

### How to use

In the current state, questions will trigger the LLM to generate a response based on the context of data already given to the LLM through declarative statements at the prompt. 

### Question/Answer BERT Model

https://sparknlp.org/2021/11/04/bert_sequence_classifier_question_statement_en.html

### Todo

* Improve chroma db query to give better/more context.
* Classifier model to distinguish imperative statements from declarative
* Chat context?