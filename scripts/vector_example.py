from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

import numpy as np
from qdrant_client.models import PointStruct

data_path = "vector.db"

client = QdrantClient(path=data_path)


client.recreate_collection(
    collection_name="my_collection",
    vectors_config=VectorParams(size=100, distance=Distance.COSINE),
)


vectors = np.random.rand(100, 100)
vectors = (vectors - 0.5) * 2
print(vectors[0])


points = []
run_id = 1

for idx, vector in enumerate(vectors):
    ps = PointStruct(
            id = idx + (run_id * 100),
            vector = vector.tolist(),
            payload = {"id": idx, "run": run_id}
        )
    points.append(ps)

client.upsert(
    collection_name="my_collection",
    points=points
)


query_vector = np.random.rand(100)
query_vector = (query_vector -0.5) * 2
hits = client.search(
    collection_name="my_collection",
    query_vector=query_vector,
    limit=5  # Return 5 closest points
)

for result in hits:
    print(result)

