import os, os.path

from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, explode
from pyspark.sql.types import StringType, ArrayType, FloatType
from sentence_transformers import SentenceTransformer
import findspark
from PyPDF2 import PdfReader
from pyspark.sql import SparkSession

from pymilvus import connections, Collection

import findspark
import re

from dotenv import load_dotenv
load_dotenv()

os.environ['PYARROW_IGNORE_TIMEZONE']='1'
os.environ['NUMEXPR_MAX_THREADS'] = '2'
os.environ['NUMEXPR_NUM_THREADS'] = '2'
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'

findspark.init()
spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext
sc.setLogLevel('ERROR')

spark.conf.set("spark.sql.shuffle.partitions", 2)


CHUNK_SIZE = 1600
CHUNK_OVERLAP = 50

# Define a UDF to extract text using PyPDF
def extract_text(file_path):
    reader = PdfReader(file_path)
    text = ''
    for i in range(0,len(reader.pages)):
        text += reader.pages[i].extract_text()
    return text


# Define the function to create embeddings
def create_embedding(text):
    # Create a SentenceTransformer model
    transformer = SentenceTransformer(os.getenv('EMBEDDING_MODEL'))
    embeddings = transformer.encode(text, convert_to_tensor=True)
    return embeddings.numpy().tolist()


def extract_text_chunks(symbol, text):
    metadata = "Document contains context of " + symbol \
        + " and is relevant to the annual reports / financial statements/ 10-K SEC fillings\n"
    chunks = []
    for i in range(0, len(text), CHUNK_SIZE):
        if i > CHUNK_OVERLAP:
            chunks.append(metadata + text[i - CHUNK_OVERLAP : i + CHUNK_SIZE])
        else:
            chunks.append(metadata + text[i : i + CHUNK_SIZE])
    return chunks

def get_stock_symbol(file_name):
    match = re.search(r'NASDAQ_([A-Z]{1,5})_2022\.pdf', file_name)
    if match:
        return match.group(1)
    return "NA"


# Register the UDF
extract_text_udf = udf(extract_text, StringType())
spark.udf.register("extract_text", extract_text_udf)

extract_text_chunks_udf = udf(extract_text_chunks, ArrayType(StringType()))
spark.udf.register("extract_text_chunks", extract_text_chunks_udf)

create_embedding_udf = udf(create_embedding, ArrayType(FloatType()))
spark.udf.register("create_embeddings", create_embedding_udf)

get_stock_symbol_udf = udf(get_stock_symbol, StringType())
spark.udf.register("get_stock_symbol", get_stock_symbol_udf)

def get_embedded_chunks(pdf_directory):
    pdf_file_paths = []
    for file in os.listdir(pdf_directory):
        if file == '.DS_Store':
            continue
        print(file)
        if file.endswith(".pdf"):
            pdf_file_paths.append(os.path.join(pdf_directory, file))
    # Create DataFrame with file paths
    pdf_files = spark.createDataFrame(pdf_file_paths, "string").toDF("file_path")
    pdf_files = pdf_files.select(
        'file_path', get_stock_symbol_udf('file_path').alias('stock_symbol'))
    # Extract text from PDF files with each line containing name of file and array of page text
    chunked_text_data = pdf_files.withColumn("text", extract_text_udf("file_path"))
    # Break text into individual row per page using explode()
    chunked_text_data = chunked_text_data.withColumn("relevant_text", \
        extract_text_chunks_udf("stock_symbol", "text"))
    # Break text into individual row per page using explode()
    chunked_text_data = chunked_text_data.select('stock_symbol', 'file_path',
        explode(chunked_text_data.relevant_text).alias('chunked_text'))
    # Convert into embeddings
    chunked_text_data = chunked_text_data.withColumn("embedded_vectors", \
        create_embedding_udf("chunked_text"))
    return chunked_text_data

def ingest_data():
    print("PDF ingestion started...")
    chunked_data = get_embedded_chunks("./rag-spark/data/annual_reports")
    # Connect to Milvus Database
    connections.connect(host=os.getenv('MILVUS_HOST'),
                        port=os.getenv('MILVUS_PORT'), secure=False)
    # Create collection if not exists
    collection_name = os.getenv('MILVUS_COLLECTION_NAME')
    collection = Collection(collection_name)
    collection.insert(chunked_data.toPandas())
    collection.flush()
    print("PDF ingestion completed...")

if __name__ == "__main__":
    ingest_data()