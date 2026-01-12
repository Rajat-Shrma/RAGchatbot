import streamlit as st
from api_utils import upload_document, list_documents
from sqlitedb.database import delete_document_record
from RAGchatbot.croma_utils import delete_doc_from_chroma

def display_sidebar():
    modeloptions = ['gemini-2.5-flash', 'gemini-2.5-pro']
    selected_model = st.sidebar.selectbox("Select Model", options = modeloptions, key='model')

    uploaded_file = st.sidebar.file_uploader("Choose a file", type=['txt', 'pdf'])

    if uploaded_file and st.sidebar.button('Upload'):
        with st.spinner("Uploading..."):
            upload_response = upload_document(uploaded_file)
            if upload_response:
                st.sidebar.success(f"File uploaded successfully with ID {upload_response['file_id']}")
                st.session_state.documents = list_documents()


    # Sidebar: List Documents
    st.sidebar.header("Uploaded Documents")
    if st.sidebar.button("Refresh Document List"):
        with st.spinner("Refreshing..."):
            st.session_state.documents = list_documents()

    if 'documents' not in st.session_state:
        st.session_state.documents = list_documents()
    documents = st.session_state.documents

    if documents:
        for doc in documents:
            st.sidebar.text(f"{doc['id']}   {doc['filename']}")

        
    delete_id = st.sidebar.number_input(
    "Enter Document ID to delete",
    min_value=1,
    step=1,
    key="delete_id"
    )

    if st.sidebar.button("Delete Doc"):
        id_deleted = delete_document_record(delete_id)
        id_deleted = delete_doc_from_chroma(delete_id)
