import json
from openai import OpenAI
from dotenv import load_dotenv
import os
import streamlit as st

load_dotenv()

OA_API_KEY = st.secrets["OPEN_AI"]
# OA_API_KEY = os.environ.get("OPEN_AI")

client = OpenAI(
    api_key=OA_API_KEY,  # This is the default and can be omitted
)

def query_retrieval(index, query_text,transaction_id,model, top_k=5):
    """
    Perform a query on Pinecone to retrieve relevant records based on vector similarity.

    Args:
        index (object): Pinecone index instance.
        query_text (str): The query text for similarity search.
        top_k (int): The number of top results to retrieve.

    Returns:
        dict: A dictionary containing query results and a summarized response.
    """
    # Step 1: Encode the query text into a vector
    query_vector = model.encode(query_text, convert_to_numpy=True).tolist()

    # Step 2: Query Pinecone for similar vectors
    query_results = index.query(
        vector=query_vector,
        top_k=top_k,
        filter={'transaction_id':transaction_id},
        include_metadata=True  # Include metadata for context
    )

    # Check if there are any results
    if not query_results.get('matches'):
        return {"error": "No relevant records found."}
    # print(query_results)
    # Step 3: Aggregate content from retrieved records
    aggregated_content = " ".join(
        match['metadata']['chunk_text'] for match in query_results['matches'] if 'chunk_text' in match['metadata']
    )
   
    record_id= []
    source=[]
    transaction_id=''
    text={}
    for idx, match in enumerate(query_results['matches']):
        transaction_id=match['id'].split('_')[0]
        metadata = match['metadata']
        record_id.append(metadata.get('record_id','').split('_')[0])
        source.append(metadata.get('source', 'unknown'))
        text[(metadata.get('source', 'unknown'),idx)]=metadata.get('chunk_text', 'unknown')
    # Step 4: Generate a response using a language model (Together Llama or OpenAI)
    input_prompt = (
        f"You are helping underwriters make decisions about policies.\n"
        f"Context: {aggregated_content}\n\n"
        f"Question: {query_text}\n"
        f"Answer:"
    )

   

    chat_completion =  client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": """You are an AI assistant designed to work with a document retrieval system for the insurance domain. The documents include emails and attachments such as Accord forms, quotations, questionnaires, and other supporting files. These files contain information such as contact details, broker details, cover quotations, location details, commercial information, risk assessments, and more Your task is to:
1. Use the document retrieval system to identify and extract relevant information based on user queries.
2. Answer queries factually and accurately by leveraging the extracted information and context stored in the vector database.
3. Provide only answer to the user's question without any other details. 
4. If the query cannot be answered with the available documents, inform the user that no relevant information was found.

Focus on providing domain-specific, contextually accurate answers in alignment with the user's query.
"""},
            {"role": "user", "content": input_prompt}
        ],

    )

    
    # Extract the response
    # print(chat_completion)
    answer=chat_completion.choices[0].message.content
    # print(answer)
    
    data={"Trasaction_id":transaction_id,
           "Record_id":list(set(record_id)),
           "Source":list(set(source)),
           "Question":query_text,
           "Answer":answer,
           "source_text":[{f"{key[0]}": value} for key, value in text.items()]

    }       
    result=json.dumps(data, indent=4) 
    return result
 

# # Example usage of the query retrieval
# query_text = "what is the subject of email?"
# response = query_retrieval(index, query_text,transaction_id="d0428c14-dc5e-4b44-8063-a2ea203d98dc")

# print(response)