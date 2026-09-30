import argparse

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, Function, FunctionType, connections, utility


COLLECTION_NAME = "hypertension_rag"
VECTOR_DIMENSION = 1024
DENSE_INDEX_PARAMS = {
    "index_type": "HNSW",
    "metric_type": "COSINE",
    "params": {"M": 16, "efConstruction": 200},
}
SPARSE_INDEX_PARAMS = {
    "index_type": "SPARSE_INVERTED_INDEX",
    "metric_type": "BM25",
    "params": {},
}


def connect(host, port):
    connections.connect(alias="default", host=host, port=port)


def build_schema():
    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=256),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=VECTOR_DIMENSION),
        FieldSchema(
            name="text",
            dtype=DataType.VARCHAR,
            max_length=65535,
            enable_analyzer=True,
            enable_match=True,
            analyzer_params={"type": "chinese"},
        ),
        FieldSchema(name="section_path", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="chunk_type", dtype=DataType.VARCHAR, max_length=32),
        FieldSchema(name="source_file", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="document_id", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="document_version", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    ]
    functions = [
        Function(
            name="text_bm25_emb",
            function_type=FunctionType.BM25,
            input_field_names=["text"],
            output_field_names=["sparse_vector"],
        )
    ]
    return CollectionSchema(
        fields=fields,
        description="Hypertension RAG chunks and table rows",
        functions=functions,
    )


def build_index_params():
    return [
        {"field_name": "vector", "index_params": DENSE_INDEX_PARAMS},
        {"field_name": "sparse_vector", "index_params": SPARSE_INDEX_PARAMS},
    ]


def ensure_collection(collection_name):
    if utility.has_collection(collection_name):
        collection = Collection(collection_name)
    else:
        collection = Collection(name=collection_name, schema=build_schema())

    indexed_fields = {index.field_name for index in collection.indexes}
    for index_config in build_index_params():
        field_name = index_config["field_name"]
        if field_name not in indexed_fields:
            collection.create_index(field_name=field_name, index_params=index_config["index_params"])

    collection.load()
    return collection


def describe_collection(collection):
    return {
        "collection": collection.name,
        "fields": [
            {
                "name": field.name,
                "type": str(field.dtype),
                "is_primary": field.is_primary,
                "params": field.params,
            }
            for field in collection.schema.fields
        ],
        "index_params": [index.params for index in collection.indexes],
        "num_entities": collection.num_entities,
    }


def main():
    parser = argparse.ArgumentParser(description="Initialize Milvus collection for hypertension RAG.")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default="19530")
    parser.add_argument("--collection", default=COLLECTION_NAME)
    args = parser.parse_args()

    connect(args.host, args.port)
    collection = ensure_collection(args.collection)
    info = describe_collection(collection)

    print(f"collection: {info['collection']}")
    print("fields:")
    for field in info["fields"]:
        print(f"  - {field['name']}: {field['type']} primary={field['is_primary']} params={field['params']}")
    print(f"index_params: {info['index_params']}")
    print(f"num_entities: {info['num_entities']}")


if __name__ == "__main__":
    main()
