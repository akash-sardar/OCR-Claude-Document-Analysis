# Typical python libraries
import os
import io
import base64
import math
from typing import List
import fitz #PyMuPDF

# GenAI libraries
import torch
from langchain_huggingface import HuggingFaceEmbeddings
import anthropic
from langchain_community.vectorstores import FAISS
from langchain_ollama.llms import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter


def llm_call(client:anthropic.Client, messages:list, model_name:str)-> str:
    """
    Call the LLM model API to process messages and return the response

    Arguments:
    - client (antrhopic.Client) LLM CLient
    - messages (list): list of prompts
    - model_name (str): Anthropic Model

    Returns:
    - response (str): The Response from the model
    """
    try:
        return client.messages.create(
            model = model_name,
            max_tokens = 8192,
            messages=messages
        ).content[0].text

    except Exception as e:
        print(f"Anthropic API error: {e}")
        return ""
        


def extract_pdf(pdf_base64:str, api_key)-> str:
    """
    Extract contents of a pdf and perform QA on it

    Arguments:
    - query (str): User query
    - pdf_base64 (str): pdf content in base64 Binary data format

    Returns:
    - response (str): 
    """
    try:
        # Create LLM client object
        client = anthropic.Client(api_key=api_key)
        # Define system prompt for PDF to text and QA
        system_prompt = """
        You are a highly skilled document parsing agent specialized in extracting structured content from complex PDFs.

        You will receive a PDF file encoded as a Base64 string. Your tasks are:

        1. Decode the Base64 input to retrieve the binary PDF.
        2. Extract **all content** from the PDF — including text, tables, headings, lists, and footnotes — **without omitting, summarizing, or rephrasing** anything.
        3. If the PDF includes **images containing text** (e.g., scanned pages), extract that text as well. OCR capability is assumed.
        4. Convert **tables** into properly formatted **Markdown tables**, preserving all data, structure, and relationships.
        5. Maintain the original document's **reading order**, formatting hierarchy, and logical structure.

        **Output Requirement**:  
        Return a single Markdown document that reflects the complete structure and contents of the original PDF, accurately and thoroughly.
        """
        model_name = "claude-sonnet-4-20250514"

        messages = [
            {
                "role":"user",
                "content":[
                    {
                        "type": "document",
                        "source":{
                            "type":"base64",
                            "media_type":"application/pdf",
                            "data": pdf_base64

                        },
                    },
                    {
                        "type": "text",
                        "text": system_prompt
                    }
                ]
            }
        ]

        # get response from LLM
        response = llm_call(client, messages, model_name)
        # print(f"Response: {response}")

        return response
    
    except Exception as e:
        print(f"PDF extraction error: {e}")
        return ""

def create_vector_store(texts: List[str])-> FAISS:
    """
    Create vector indexes on documents for faster retrieval

    Arguments:
    - texts (List(str)): chunked documents
    Returns:
    - vector_store (FAISS index) - embeddings of chunked documents
    """
    try:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Device: {device}")
    
        # Load sentence transformer model
        embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-mpnet-base-v2')
        vector_store = FAISS.from_texts(texts, embeddings)
        return vector_store
    except Exception as e:
        print(f"Error encountered while creating vector store: {e}")
        raise (e)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
        

def create_question_answer_chain(vector_store):

    try:
        llm = OllamaLLM(model = "llama2:latest")

        if not llm:
            raise (f"Ollama error")
        
        # Create custom prompt template
        prompt_template = """
            Use the following pieces of context to answer the question at the end. 
            Check context very carefully and reference and try to make sense of that before responding.
            If you don't know the answer, just say you don't know. 
            Don't try to make up an answer.
            Answer must be to the point.
            Think step-by-step.
            
            Context: {context}
            
            Question: {question}
            
            Answer:"""
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        # Create LCEL chain
        retriever = vector_store.as_retriever(search_kwargs = {"k":3})

        question_answer_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | PROMPT
            | llm
            | StrOutputParser()
        )

        return question_answer_chain

    except Exception as e:
        raise (f"Error while creating LCEL: {e}")


def read_pdf_in_pairs(file_path = str)->list:
    """
    Read PDF two pages at a time and convert them to binary
    Arguments:
    file_path (str) : File Path of the PDF to be extracted
    Returns:
    - List of encoded documents
    """
    # Read the PDF
    doc = fitz.open(file_path)
    total_pages = len(doc)
    number_of_pairs = math.ceil(total_pages/2)

    base_64_content = []

    # Read pairs and encode them to binary
    for pair_num in range(number_of_pairs):
        start_index = pair_num * 2
        end_index = min(start_index+2, total_pages)

        # create temp docuement
        output_doc = fitz.open ()

        for page_index in range(start_index, end_index):
            output_doc.insert_pdf(doc, from_page = page_index, to_page = page_index)
        
        # save pairs as a byte stream in memory, instead of writing it to a file on disk.
        # base64 needs a bytes stream not file path
        output_buffer = io.BytesIO()
        output_doc.save(output_buffer) # saving output_doc as byte stream
        pair_content = output_buffer.getvalue() # binary representation of the PDF pair

        # Convert to base64 binary and store in results
        base_64_content.append(base64.b64encode(pair_content).decode("utf-8"))

        # Close the temprary document
        output_doc.close()
        output_buffer.close()
    
    doc.close()
    
    return base_64_content


def main(key):
    try:
        ANTHROPIC_API_KEY = key
       
        os.makedirs("Data", exist_ok=True)
        extracted_file_path = "Data"+"\extracted_pdf.txt"
        try:
            if os.path.exists(extracted_file_path):
                print(f"Reading cached extracted pdf")
                with open(extracted_file_path, 'r') as file:
                    structured_content = file.read()
            else:
                # Step 1: Read the PDF and convert it to binary
                file_path = "Data"+"\sample-tables.pdf"
                base_64_content = read_pdf_in_pairs(file_path)

                # Step 2: Extract PDF content using Claude
                structured_content = ""
                for content in base_64_content:
                    structured_content += "\n"+extract_pdf(content, ANTHROPIC_API_KEY)                
        except FileNotFoundError:
            raise (f"Cached extracted file not found")
        except Exception as e:
            print(f"An error occurred while reading cached file: {e}")
        
        if not structured_content:
            raise ("PDF not extracted")
        
        # Sanitize extracted content
        structured_content = (
            structured_content
            .replace("☑", "[Yes]")
            .replace("☒", "[No]")  # Or "[X]" or "❌"
            .replace("☐", "[None]")
            .replace("✓", "[Yes]")  # safer printable form
            .replace("✗", "[No]")
        )        
        with open(extracted_file_path, "w", encoding="utf-8") as file:
            file.write(structured_content)

        # Step 3: Chunk extracted document
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 2000,
            chunk_overlap = 200,
            is_separator_regex= False
        )
        text_chunks = text_splitter.split_text(structured_content)

        # Step 4: Create vector store with embeddings
        vector_store = create_vector_store(text_chunks)

        # Step 5: Create QA chain and get response of questions
        qa_chain = create_question_answer_chain(vector_store)

        # Step 6: Execute and test Question/Answer
        query = "How did Respondent B respond to the question that if he is a UK citizen or not ?"
        response = qa_chain.invoke(query)
        print(f"\nAnswer: {response}")

    except Exception as e:
        raise ("Error found in Main : {e}")
    

if __name__ == "__main__": 
    main()
    



