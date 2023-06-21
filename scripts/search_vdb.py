import json
from PIL import Image, ImageDraw
import numpy as np
import dlib
import face_recognition_models as frm

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

def detect_face(file_path):

    arr = dlib.load_rgb_image(file_path)

    face_detector = dlib.get_frontal_face_detector()
    face_at = face_detector(arr, 1)

    if len(face_at) == 0:
        return None
    elif len(face_at) > 1:
        raise Exception("More than one face detected, for encoding only one face per image is allowed : " + file_path)
    
    predictor_5_point_model = frm.pose_predictor_five_point_model_location()
    pose_predictor_5_point = dlib.shape_predictor(predictor_5_point_model)
    face_data = pose_predictor_5_point(arr, face_at[0])

    model_location = frm.face_recognition_model_location()
    face_encoder = dlib.face_recognition_model_v1(model_location)
    enc_data = face_encoder.compute_face_descriptor(arr, face_data, 1)

    dib_face_enc = np.array(enc_data)
    return dib_face_enc.tolist()


# extract face encoding data
image_path = "..\\data\\maxresdefault.jpg"
enc = detect_face(image_path)

# create vector db client
data_path = "vector.db"
q_client = QdrantClient(path=data_path)

# search for a face match in our vector DB using the face encoded vector
hits = q_client.search(
    collection_name="face_lookup",
    query_vector=enc,
    limit=6
)
for result in hits:
    #print(result)
    print("%s\t%s\t%s" % (result.id, result.score, result.payload["name"]))

