import os
from llama_index.llms import Replicate
from llama_index import set_global_tokenizer
from transformers import AutoTokenizer
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index import ServiceContext
from llama_index import VectorStoreIndex, SimpleDirectoryReader


class TextGenerator:
    def __init__(self, config):
        print("Started process")
        os.environ["REPLICATE_API_TOKEN"] = config['API_KEY']
        self.llm = Replicate(
            model=config['replicate_config']['model'],
            temperature=config['replicate_config']['temperature'],
            additional_kwargs=config['replicate_config']['additional_kwargs']
        )
        # set tokenizer to match LLM
        set_global_tokenizer(
            AutoTokenizer.from_pretrained(config['tokenizer']).encode
        )
        embed_model = HuggingFaceEmbedding(model_name=config['embed_model'])
        service_context = ServiceContext.from_defaults(
            llm=self.llm, embed_model=embed_model
        )
        documents = SimpleDirectoryReader("data").load_data()
        index = VectorStoreIndex.from_documents(
            documents, service_context=service_context
        )
        print("Loaded data")
        self.query_engine = index.as_query_engine()
        print("Intialized query engine")

    def query(self, query_str):
        response = self.query_engine.query(query_str)
        res_text = response.response.split('\n')
        return res_text

