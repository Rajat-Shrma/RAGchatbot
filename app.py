from fastapi import FastAPI,UploadFile,File, HTTPException
from CONSTS import retriever
from sqlitedb.database import get_chat_history, insert_application_logs
from RAGchatbot.chatting import contextualize_chain, rag_chain
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel,Field
from typing import Annotated, List
import uuid
from RAGchatbot.croma_utils import index_document_to_chroma, delete_doc_from_chroma
from sqlitedb.database import insert_document_record, delete_document_record, get_all_documents
import os
from datetime import datetime

class chat_input(BaseModel):

    question : Annotated[str, Field(description="Enter your question")]
    session_id: Annotated[str, Field(default=None, description="your session id")]
    model : Annotated[str, Field(default='gemini-2.5-flash')]

class response_model(BaseModel):
    answer : str 
    session_id : str
    model : str

class DocumentInfo(BaseModel):
    id: int
    filename: str
    upload_timestamp: datetime

class DeleteFileRequest(BaseModel):
    file_id: int

app = FastAPI()


@app.get("/")
def welcome():
    return "Your Welcome"

@app.post("/upload")
def uploading_document_indexing_doc(file: UploadFile = File(...)):
    temp_file_path = f"temp_{file.filename}"

    try:
        file_id = insert_document_record(file.filename)
        success = index_document_to_chroma(file, file_id)

        if success:
            return {'message': f"File {file.filename} has been successfully uploaded and indexed.", 'file_id': file_id}
        else:
            delete_document_record(file_id)

            raise HTTPException(status_code=500, details = f"Failed to index {file.filename}.")
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
                  


@app.post("/chat", response_model=response_model)

def chat(data : chat_input):
    question = data.question
    session_id = data.session_id
    model = data.model

    if not session_id:
        session_id = str(uuid.uuid4())

    chat_history = get_chat_history(session_id)

    answer = rag_chain.invoke({
        "input": question,
        "chat_history": chat_history
    })

    insert_application_logs(session_id,question, answer, model)

    return response_model(answer=answer, session_id=session_id, model=model)    

    
@app.get("/list-docs", response_model=List[DocumentInfo])
def list_documents():
    return get_all_documents()



@app.post("/delete-doc")
def delete_document(request: DeleteFileRequest):
    # Delete from Chroma
    chroma_delete_success = delete_doc_from_chroma(request.file_id)

    if chroma_delete_success:
        # If successfully deleted from Chroma, delete from our database
        db_delete_success = delete_document_record(request.file_id)
        if db_delete_success:
            return {"message": f"Successfully deleted document with file_id {request.file_id} from the system."}
        else:
            return {"error": f"Deleted from Chroma but failed to delete document with file_id {request.file_id} from the database."}
    else:
        return {"error": f"Failed to delete document with file_id {request.file_id} from Chroma."}
