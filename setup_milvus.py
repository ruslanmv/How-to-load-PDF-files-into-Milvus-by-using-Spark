import os
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from dotenv import load_dotenv
load_dotenv()

def init_vectordb():
    HOST = os.getenv('MILVUS_HOST')
    PORT = os.getenv('MILVUS_PORT')
    # Connect to Milvus Database
    connections.connect(host=HOST, port=PORT, secure=False)

    # Create collection if not exists
    collection_name = os.getenv('MILVUS_COLLECTION_NAME')

    # Remove collection if it already exists (only for test)
    if utility.has_collection(collection_name):
        print('Dropping existing collection "%s"' % collection_name)
        utility.drop_collection(collection_name)

    # Create collection which includes the id, title, and embedding.
    fields = [
        FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name='stock_symbol', dtype=DataType.VARCHAR, max_length=10),
        FieldSchema(name='file_path', dtype=DataType.VARCHAR, max_length=200),
        FieldSchema(name='chunked_text', dtype=DataType.VARCHAR, max_length=2200),
        FieldSchema(name='embedded_vectors', dtype=DataType.FLOAT_VECTOR, dim=384)
    ]
    
    print('Creating collection and index for "%s"' % collection_name)
    schema = CollectionSchema(fields=fields)
    collection = Collection(name=collection_name, schema=schema)
    # Create an IVF_FLAT index for collection.
    index_params = {
        'metric_type':'L2',
        'index_type':"IVF_FLAT",
        'params':{"nlist":768}
    }
    collection.create_index(field_name="embedded_vectors", index_params=index_params)
    collection.load()
    return collection

if __name__ == '__main__':
    init_vectordb()