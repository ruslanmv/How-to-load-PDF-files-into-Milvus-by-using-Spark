import os
import gradio as gr
from pymilvus import connections, Collection
import replicate
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
os.environ["REPLICATE_API_TOKEN"] = os.getenv('REPLICATE_API_TOKEN')
SYSTEM_TEMPLATE = os.getenv('SYSTEM_TEMPLATE')
collection_name = os.getenv('MILVUS_COLLECTION_NAME')

#'''
# Search the database based on input text
def embed_search(data):
    # Create a SentenceTransformer model
    transformer = SentenceTransformer(os.getenv('EMBEDDING_MODEL'))
    embeds = transformer.encode(data)
    return [x for x in embeds]

def data_querying(input_text):
    print('Searching in vector DB ...')
    search_terms = [input_text]
    search_data = embed_search(input_text)
    # Connect to Milvus Database
    connections.connect(host=os.getenv('MILVUS_HOST'), port=os.getenv('MILVUS_PORT'), secure=False)
    collection = Collection(collection_name)
    response = collection.search(
        data=[search_data],  # Embedded search value
        anns_field="embedded_vectors",  # Search across embeddings
        param={},
        limit = 3,  # Limit to top_k results per search
        output_fields=['chunked_text', 'file_path']  # Include required field in result
    )
    
    prompt_text = ''
    for _, hits in enumerate(response):
        for hit in hits:
            prompt_text += hit.entity.get('chunked_text') + '\n\nSOURCE: ' + hit.entity.get('file_path') + '\n\n'

    output = replicate.run(
        os.getenv("LLAMA2_LLM_MODEL"),
        input={
            "system_prompt": os.getenv('LLAMA2_SYSTEM_PROMPT'),
            "prompt": 'Context: ' + prompt_text[:4096] + '\n' + 'Question: ' + search_terms[0],
            "temperature": 0.5,
            "max_length": 3000,
            "top_p": 1
        }
    )
    llm_response = ''
    for item in output:
        llm_response += item
        yield llm_response
    yield llm_response


iface = gr.Interface(fn=data_querying,
                     inputs=gr.components.Textbox(lines=7, label="Enter your question", ),
                     outputs="text",
                     title="Financial Knowledge Base",
                     description="Ask a question about the NASDAQ data and get a response",
                     ).queue()

iface.launch(share=False)