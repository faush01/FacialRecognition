from PIL import Image, ImageDraw
import dlib


image_path = "..\\data\\training\\Gal Gadot\\03.jpg"

arr = dlib.load_rgb_image(image_path)

face_detector = dlib.get_frontal_face_detector()
face_at = face_detector(arr, 1)

print("Faces found : " + str(len(face_at)))
for face_box in face_at:
    print(" - %s" % (face_box))

# save detection image
im = Image.fromarray(arr.astype('uint8'), "RGB")
img1 = ImageDraw.Draw(im)
for face_box in face_at:
    shape = [(face_box.left(), face_box.top()), (face_box.right(), face_box.bottom())]
    img1.rectangle(shape, outline="red", width=3)
im.save(image_path + ".box.jpg")
print("image saved")
