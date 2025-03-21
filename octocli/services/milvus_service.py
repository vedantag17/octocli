from pymilvus import (
    connections,
    Collection,
    CollectionSchema,
    FieldSchema,
    DataType,
    utility
)
from typing import Dict, List
import json

class MilvusService:
    def __init__(self, host: str = "localhost", port: int = 19530):
        self.host = host
        self.port = port
        self.connect()
        self._init_collections()

    def connect(self):
        """Connect to Milvus server"""
        try:
            connections.connect(host=self.host, port=self.port)
        except Exception as e:
            raise Exception(f"Failed to connect to Milvus: {str(e)}")

    def _init_collections(self):
        """Initialize collections for storing repository data"""
        # Repository metadata collection
        self.repo_schema = CollectionSchema(
            fields=[
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=200),
                FieldSchema(name="description", dtype=DataType.VARCHAR, max_length=2000),
                FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=10000),
            ],
            description="Repository metadata"
        )

        if not utility.has_collection("repository"):
            self.repo_collection = Collection(
                name="repository",
                schema=self.repo_schema,
                using='default'
            )

    def store_metadata(self, metadata: Dict):
        """Store repository metadata in Milvus"""
        try:
            self.repo_collection.insert([
                [metadata.name],
                [metadata.description],
                [json.dumps(metadata.dict())]
            ])
            self.repo_collection.flush()
        except Exception as e:
            raise Exception(f"Failed to store metadata: {str(e)}")
