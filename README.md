# Everybody's doing it, so why shouldn't I?
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

```conda create -f ./environment.yml```

### Example responses

#### Response 1

> I am a software developer with nine years of experience, and have worked in various roles throughout the industry. I am currently a full-stack developer for a Fortune 50 company, and previously worked as an Associate Team Lead.
>
> I have experience working in various industries, including energy and finance. I have also worked on large, complex projects with teams of varying sizes. I am familiar with multiple technologies and programming languages, including C#, JavaScript, Python, SQL, and more.
>
> My strengths include my ability to quickly learn new technologies, as well as my ability to work independently or in a team setting. I have strong communication skills and the ability to effectively communicate with both technical and non-technical stakeholders. I am also proficient in agile development methodologies, including scrum and kanban.
>
> In summary, I am an experienced software developer with a wide range of experience and skills. I am confident that my qualifications make me a strong candidate for the position you are advertising. Thank you for considering my application.

#### Response 2

> I have developed an extensive list of technical skills and experience over the course of my career in software development. I am a hard worker with a proven track record of delivering high-quality software on time. I am also a strong team leader who is able to manage multiple projects simultaneously.
>
> My resume includes a variety of positions in different industries, including: 
>
> • Lead Software Developer (Spot Traceability, Remote) 
>
> • Associate Team Lead (JPMorgan Chase & Co., Houston) 
>
> • Machine Learning Engineer (Chevron, Houston)
> 
> These experiences have given me the knowledge and skills to be a successful software engineer in your company. I am familiar with many different programming languages and frameworks, including JavaScript, Python, Java, C#, .NET Framework, etc. I have also developed a strong understanding of the latest technologies such as Big Data, Machine Learning, DevOps, Continuous Integration and Delivery, and Cloud Computing.
>
> Additionally, I have experience working with various data structures and algorithms. I am able to analyze problems and develop solutions efficiently.
> I am confident that I can make a positive impact on your company and look forward to the opportunity to discuss my qualifications further.

# Question/Answer BERT Model

https://sparknlp.org/2021/11/04/bert_sequence_classifier_question_statement_en.html