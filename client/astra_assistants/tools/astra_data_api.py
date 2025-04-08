import json
import logging
import os

from openai import OpenAI

from .tool_interface import ToolInterface
from astrapy import DataAPIClient


logger = logging.getLogger(__name__)

class AstraDataAPITool(ToolInterface):
    def __init__(self, db_url: str, collection_name, vectorize: bool = False, openai_client: OpenAI = None, embedding_model:str = None, namespace: str = None):
        client = DataAPIClient()
        db = client.get_database(
            db_url,
            token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
            keyspace=namespace,  # passing None is on (same as omitting the parameter)
        )
        self.collection = db.get_collection(collection_name)
        self.vectorize = vectorize
        self.openai_client = openai_client
        self.embedding_model = embedding_model

    def call(self, arguments):
        query = arguments['arguments']
        try:
            if self.vectorize:
                results = self.collection.find(
                    sort={"$vectorize": query},
                    limit=2,
                    projection={"$vectorize": True},
                    include_similarity=True,
                )
                print(f"Vector search results for '{query}':")
                result_array = []
                for result in results:
                    result_array.append(result)
                return json.dumps(result_array)
            else:
                assert self.openai_client is not None and self.embedding_model is not None, "OpenAI client and embedding_model are required for non-vectorized search."
                vector = self.openai_client.embeddings.create(input=query, model=self.embedding_model).data[0].embedding
                results = self.collection.find(
                    sort={"$vector": vector},
                    limit=10,
                    projection={"$vector": False},
                )
                print(f"Vector search results for '{query}':")
                result_array = []
                for result in results:
                    result_array.append(result)
                return json.dumps(result_array)
        except Exception as e:
            logger.error(e)
            raise e