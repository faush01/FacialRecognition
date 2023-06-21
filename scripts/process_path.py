import json
import os
from os import walk
from PIL import Image, ImageDraw
import numpy as np
import dlib
import face_recognition_models as frm


img_exts = [".jpg", ".png", ".jpeg", ".gif"]


def check_file_name(filename):
    if ".box." in filename:
        return False
    lcn = filename.lower()
    for ext in img_exts:
        if lcn.endswith(ext):
            return True
    return False


def find_images(base_path, found_images):
    for (dirpath, dirnames, filenames) in walk(base_path):
        for file_name in filenames:
            if check_file_name(file_name):
                found_images.append(os.path.join(dirpath, file_name))


def write_img_meta_data(enc_data, img_path):
        img_meta_data = {}
        data_path = img_path + ".json"
        if os.path.exists(data_path):
            with open(data_path, "r") as json_file:
                img_meta_data = json.load(json_file)
        
        img_meta_data["encoded"] = enc_data
        # get the name from the path
        img_meta_data["name"] = img_path.split(os.path.sep)[-2]

        json_data = json.dumps(img_meta_data)
        with open(data_path, "w") as json_file:
            json_file.write(json_data)


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


#
# walk path and create encodings
#

# get all the images in the training path
training_path = "..\\data\\training"
image_files = []
find_images(training_path, image_files)

# process images and write out 
for image_path in image_files:
    print ("Processing image : " + image_path)

    face_enc = detect_face(image_path)
    if face_enc:
        write_img_meta_data(face_enc, image_path)
