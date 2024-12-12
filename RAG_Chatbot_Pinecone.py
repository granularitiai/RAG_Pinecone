#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install --upgrade --quiet  langchain-pinecone langchain-openai langchain')


# In[ ]:


get_ipython().system('pip install -U langchain-pinecone')


# In[ ]:


get_ipython().system('pip install pinecone-client')


# In[51]:


import os
import openai
import langchain
import pinecone 
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore as PVC
from langchain.llms import OpenAI


# In[52]:


from dotenv import load_dotenv
load_dotenv()


# In[70]:


def load_document(file):
    name, extension = os.path.splitext(file) 
    if extension == '.html':
        from langchain.document_loaders import UnstructuredHTMLLoader
        print(f'load {file}...')
        loader = UnstructuredHTMLLoader(file)
    elif extension == '.txt':
        from langchain.document_loaders import TextLoader  
        print(f'load {file}...')
        loader = TextLoader(file)
    elif extension == '.pdf':
        from langchain.document_loaders import PyPDFLoader
        print(f'load {file}...')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain.document_loaders import Docx2txtLoader
        print(f'load {file}...')
        loader = Docx2txtLoader(file)
    else:
        print('The document format is not supported!')
        return None

    data = loader.load()
    return data


# In[71]:


document = "document_path"
content = load_document(document)
len(content)


# In[72]:


def split (data, chunk_size=800):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=50)
    fragments = text_splitter.split_documents(data)
    return fragments


# In[73]:


fragments = split(content)
print(len(fragments))


# In[74]:


os.environ['OPENAI_API_KEY'] = "OPENAI_API_KEY"
os.environ['PINECONE_API_KEY'] = "PINECONE_API_KEY"


# In[75]:


embeddings = OpenAIEmbeddings(api_key = os.environ['OPENAI_API_KEY'])
embeddings


# In[76]:


vectors = embeddings.embed_query("How are you?")
len(vectors)


# In[77]:


from pinecone import Pinecone

pc = Pinecone(api_key= 'PINECONE_API_KEY')
index_name = "serverless-index"


# In[78]:


Index = PVC.from_documents(content, embeddings, index_name = index_name)


# In[79]:


def retrieve_query(query, k=2):
    matching_results = Index.similarity_search(query, k=k)
    return matching_results


# In[80]:


from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI


# In[81]:


llm = OpenAI(model_name = "davinci-002", temperature = 0.1)
chain = load_qa_chain(llm, chain_type = "map_rerank")


# In[82]:


def retrieve_answer(query):
    doc_search = retrieve_query(query)
    print(doc_search)
    response = chain.run(input_documents = doc_search, question=query)
    return response


# In[83]:


our_query = "EnterYourOwnQuery"
answer = retrieve_answer(our_query)
print(answer)


# In[ ]:




