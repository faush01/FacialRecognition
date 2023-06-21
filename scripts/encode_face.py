import json
from PIL import Image, ImageDraw
import numpy as np
import dlib
import face_recognition_models as frm


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


image_path = "..\\data\\training\\Gal Gadot\\03.jpg"
enc = detect_face(image_path)
print(enc)

enc_data = {}
enc_data["encoded"] = enc

json_data = json.dumps(enc_data)
with open(image_path + ".json", "w") as json_file:
    json_file.write(json_data)

