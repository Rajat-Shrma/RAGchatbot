from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from CONSTS import model
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from CONSTS import retriever
from RAGchatbot.utility import format_docs

contextualize_q_system_prompt = '''
Given a chat history and the latest user question.
which might reference context in the chat history.
Formulate a standalone question which can be understood without
the chat history. DO NOT ANSWER the question, 
just reformulate it if needed and otherwise return as it is.
'''

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ('system', contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ('human',"Question : {input}")
])

contextualize_chain = contextualize_q_prompt | model | StrOutputParser()

system_message = """
You are a Government Scheme Advisor.

Your role:
- Answer the user’s question using ONLY the provided context.
 ### Available Context:
    {context}
- Do NOT use prior knowledge, assumptions, or external information.
- Respond in clear, simple English suitable for the general public.

Strict rules:
1. If you are not able to frame answer from the given context reply-
   "The requested information is not available in the provided documents."
2. Do NOT create or assume new schemes, rules, amounts, eligibility criteria, or procedures.
3. Do NOT mix information from different schemes.
4. Keep answers factual, structured, and easy to understand.

Answer style:
- Short sentences
- Bullet points or numbered steps
- No technical or legal jargon unless present in the context

"""

human_message = '''
### User Question:
{question}

If the question is unclear or ambiguous:
- Politely ask the user to clarify

If the information is not available:
- Follow the system rule and clearly state that it is not available
'''

qa_prompt  = ChatPromptTemplate([
    ('system', system_message),
    MessagesPlaceholder("chat_history"),
    ('human',human_message)
])


retrieval_chain = (
    contextualize_chain
    | retriever
    | format_docs
)


rag_chain = (
    {
        "question": lambda x: x["input"],
        "context": retrieval_chain,
        "chat_history": lambda x: x['chat_history']
    }
    | qa_prompt
    | model
    | StrOutputParser()
)

if __name__=='__main__':
    ans = rag_chain.invoke({'input': "what is ayushman bharat yojana", 'chat_history': []})
    print(ans)

