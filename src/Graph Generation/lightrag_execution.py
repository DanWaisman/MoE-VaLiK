import asyncio
import os
import inspect
import logging
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

import networkx as nx
from pyvis.network import Network

import json

### -------------------------------------------------------------------------------------- ###

# this is an asynchronous function, it's designed to run without blocking the rest of the program.
async def print_stream(stream):
    async for chunk in stream:
        print(chunk, end="", flush=True)

### -------------------------------------------------------------------------------------- ###

WORKING_DIR = "./KGs_generated/ScienceQA_Text_and_Image_R1_v4" # directory where the knowledge graph will be stored
graph_file_path = './KGs_generated/ScienceQA_Text_and_Image_R1_v4/graph_chunk_entity_relation.graphml'

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

# creates working directory if it does not exist
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# setting up the environment
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    llm_model_name="deepseek-r1:7b",
    llm_model_max_async=160,
    llm_model_max_token_size=65536,
    llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 65536}},
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: ollama_embed(
            texts, embed_model="nomic-embed-text", host="http://localhost:11434"
        ),
    ),
)

# defines path for image_descriptions + test linked to each image
directory = './datasets/ScienceQA/data/scienceqa/images/mac_test'
original_text_path = './src/Original_Text_Compilation/ScienceQA_Text.txt'

print(f'\nModel in use: {rag.llm_model_name}\n')

# --------------------- inserting files with image descriptions --------------------- #
print(f'Inserting image descriptions\n')
i=0

caption_string = ""
for root, dirs, files in os.walk(directory):
    for filename in files:
        if filename.endswith('_filtered.txt'):
            i+=1

            file_path = os.path.join(root, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()

            print('==========================================================')
            print(f"Inserting description from Image {i} - ({file_path} )")
            # inserting image descriptions into knowledge graph
            # adding image path as metadata (to make it Multi Modal)
            rag.insert(
                text, 
                split_by_character='.'                       #  '\n\n' , '\n', '.', 
            )

print("Finished inserting all documents.")

### ---------------------------- querying the knowlegde graph ---------------------------- ###
# naive
print('QUERIES...')
naive = rag.query("What are the top themes in this story?", param=QueryParam(mode="naive"))

# local
local = rag.query("What are the top themes in this story?", param=QueryParam(mode="local"))

# global
global_ = rag.query("What are the top themes in this story?", param=QueryParam(mode="global"))

# hybrid
hybrid = rag.query("What are the top themes in this story?", param=QueryParam(mode="hybrid"))

# stream response (output is sent as its being generated, rather than waiting for the full generation before returning)
resp = rag.query( "What are the top themes in this story?",
                  param=QueryParam(mode="hybrid", stream=True),)

# logic checks what kind of response the rag.query() method returned and chooses the correct way to print it
# For handling 'stream' responses
if inspect.isasyncgen(resp):
    asyncio.run(print_stream(resp))
else:
    print(resp)

# ============================================================================================ #