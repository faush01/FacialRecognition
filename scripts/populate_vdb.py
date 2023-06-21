import os
from os import walk
import json

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

import numpy as np
from qdrant_client.models import PointStruct


def find_meta_files(base_path, found_images):
    for (dirpath, dirnames, filenames) in walk(base_path):
        for file_name in filenames:
            if file_name.endswith(".json"):
                found_images.append(os.path.join(dirpath, file_name))


def load_meta_data(meta_path):
    with open(meta_path, "r") as json_file:
        meta_data = json.load(json_file)
    return meta_data   


def get_qdrant_client():
    data_path = "vector.db"
    client = QdrantClient(path=data_path)

    client.recreate_collection(
        collection_name = "face_lookup",
        vectors_config = VectorParams(size=128, distance=Distance.COSINE)
    )
    return client


# qdrent client
q_client = get_qdrant_client()

# process files
base_data_path = "..\\data\\training"
meta_files_found = []
meta_files = find_meta_files(base_data_path, meta_files_found)

# build vectors to store
vector_points = []
for idx, meta_file in enumerate(meta_files_found):
    print(str(idx) + " - " + meta_file)
    meta_data = load_meta_data(meta_file)
    # print(meta_data)
    ps = PointStruct(
            id = idx,
            vector = meta_data["encoded"],
            payload = {"name": meta_data["name"]}
        )
    vector_points.append(ps)    


# add face encoded vectors to the db
q_client.upsert(
    collection_name="face_lookup",
    points=vector_points
)

# test query
search_meta_data = load_meta_data(meta_files_found[5])
hits = q_client.search(
    collection_name="face_lookup",
    query_vector=search_meta_data["encoded"],
    limit=6
)
for result in hits:
    #print(result)
    print("%s\t%s\t%s" % (result.id, result.score, result.payload["name"]))

