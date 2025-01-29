from pyzerox import zerox
import os
import json
import asyncio

import email
from email import policy
from email.parser import BytesParser
from io import BytesIO
from pdf2image import convert_from_bytes

import base64
import os
import uuid
import itertools
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import TokenTextSplitter
from dotenv import load_dotenv

load_dotenv()


# Pinecone initialization
open_ai_cred = os.environ.get("OPEN_AI")
os.environ["OPENAI_API_KEY"] = open_ai_cred


# Global counter for unique chunk IDs
global_counter = itertools.count()


# Function to parse email and extract content and attachments
def parse_email(email_file):
    email = BytesParser(policy=policy.default).parse(email_file)
    subject = email['subject'] or ''
    sender = email['from'] or ''
    recipient = email['to'] or ''
    body = ""
    attachments = []

    for part in email.walk():
        if part.get_content_type() == 'text/plain':
            body += part.get_payload(decode=True).decode(part.get_content_charset(), errors="replace")
        elif part.get_content_disposition() == "attachment":
            attachments.append(part)
    
    email_text = f"Subject: {subject}\nFrom: {sender}\nTo: {recipient}\n\n{' '.join(body.split())}"
    return {'email_text': email_text.strip(), 'email_name': subject or 'Unnamed Email', 'attachments': attachments}

# Function to encode image as base64
def encode_image(image):
    with BytesIO() as output:
        image.save(output, format="PNG")
        return base64.b64encode(output.getvalue()).decode("utf-8")


# Function to process PDF attachments using Together Llama model
async def process_pdf_attachment(part, model_z):
    content_disposition = part.get("Content-Disposition", "")
    filename=None
    kwargs = {}
    extracted_text=''
    if "attachment" in content_disposition and part.get_filename()[-3:]=="pdf":
        filename = part.get_filename()
        pdf_data = part.get_payload(decode=True)
        try:
            os.makedirs('temp_data', exist_ok=True)
        except:
            pass
        save_path = os.path.join('temp_data', filename)

        with open(save_path, "wb") as f:
            f.write(pdf_data)
        select_pages = None ## None for all, but could be int or list(int) page numbers (1 indexed)

        output_dir = "./output_test" ## directory to save the consolidated markdown file
        result = await zerox(file_path=save_path, model=model_z, output_dir=None,
                            custom_system_prompt=None,select_pages=select_pages, **kwargs)
        

        for page in result.pages:
        
            extracted_text+=page.content
            extracted_text+='\n\n'

    return filename,extracted_text

    

def token_chunk(text):
    

    text_splitter = TokenTextSplitter(
        chunk_size=100,
        chunk_overlap=10
    )
    chunks = text_splitter.split_text(text)
    return chunks


# Function to upsert chunks to Pinecone
def upsert_to_pinecone(chunks, transaction_id, index, record_id, source_name,model):
    """Upsert the chunks into Pinecone with metadata"""
    for chunk in chunks:
        unique_id = next(global_counter)  # Get a globally unique count value
        unique_transaction_id = f"{transaction_id}_chunk{unique_id}"
        embedding = model.encode(chunk, convert_to_numpy=True).tolist()
        metadata = {"source": source_name, "chunk_text": chunk, 'record_id': record_id,'transaction_id':transaction_id}
        index.upsert([(unique_transaction_id, embedding, metadata)])

# Function to embed and insert email data into Pinecone
async def embed_and_insert_email_data(email_file_path, index,model,model_z):
    # Parse the email
    with open(email_file_path, "rb") as email_file:
        email_data = parse_email(email_file)

    transaction_id = str(uuid.uuid4())  # Generate unique transaction ID

    # Process email body
    # email_chunks = token_chunk(email_data['email_text'])
    email_record_id = str(uuid.uuid4())  # Unique record ID for email body
    upsert_to_pinecone([email_data['email_text']], transaction_id, index, email_record_id, 'Email',model=model)

    prev_filename = None
    for attachment in email_data['attachments']:
        filename, attachment_text = await process_pdf_attachment(attachment,model_z= model_z)
        if filename:
            # Generate a new record ID only if the filename changes
            if filename != prev_filename:
                attachment_record_id = str(uuid.uuid4())
                prev_filename = filename
            attachment_chunks = token_chunk(attachment_text)
            upsert_to_pinecone(attachment_chunks, transaction_id, index, attachment_record_id, filename,model=model)
        
        
            
    return transaction_id

