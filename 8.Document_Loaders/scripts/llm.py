from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.prompts import (SystemMessagePromptTemplate,HumanMessagePromptTemplate,
                                    ChatPromptTemplate,PromptTemplate)
from langchain_core.output_parsers import StrOutputParser

from dotenv import load_dotenv
load_dotenv()


base_url = "http://localhost:11434"
model = 'llama3.2'

ollama_llm = ChatOllama(base_url=base_url,model=model)
groq_llm = ChatGroq(model="llama-3.2-3b-preview")

system = SystemMessagePromptTemplate.from_template("""You are helpful AI assistant who answer user question based on the provided context.""")

prompt = """Answer user question based on the provided context ONLY! If you do not know the answer, just say "I don't know".
            ### Context:
            {context}

            ### Question:
            {question}

            ### Answer:"""

prompt = HumanMessagePromptTemplate.from_template(prompt)

messages = [system, prompt]
template = ChatPromptTemplate(messages)

print(template)
# template.invoke({'context': context, 'question': "How to gain muscle mass?", 'words': 50})

# qna_chain = template | ollama_llm | StrOutputParser()
qna_chain = template | groq_llm | StrOutputParser()

def ask_llm(context,question):
    return qna_chain.invoke({'context':context,'question':question})