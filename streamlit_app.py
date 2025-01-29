import streamlit as st
import requests

FASTAPI_URL = "http://127.0.0.1:8000"  # FastAPI endpoint

def upload_email(file):
    response = requests.post(f"{FASTAPI_URL}/upload-email/", files={"file": file})
    return response.json()

def query_pinecone(query, transaction_id=None):
    payload = {"query_text": query}
    if transaction_id:
        payload["transaction_id"] = transaction_id
    response = requests.post(f"{FASTAPI_URL}/query-pinecone/", json=payload)
    return response.json()

# Streamlit UI
st.title("Email Attachment Processing")

# Section for uploading email
st.subheader("Upload an Email to Process")

uploaded_file = st.file_uploader("Choose a file", type=["eml"])
if uploaded_file is not None:
    # Upload email to FastAPI
    response = upload_email(uploaded_file)
    st.write(response)
    transaction_id = response.get("transaction_id")

    if transaction_id:
        st.success("Email successfully uploaded and processed.")
        st.session_state.transaction_id = transaction_id

# Section for querying Pinecone
st.subheader("Query Pinecone")
query = st.text_input("Enter query")

if query:
    transaction_id = st.session_state.get("transaction_id")
    response = query_pinecone(query, transaction_id)
    st.json(response)

else:
    st.warning("Please enter a query to search Pinecone.")
