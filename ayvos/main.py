import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from PIL import Image  
import PIL 
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import keras
from mtcnn.mtcnn import MTCNN
import keras_vggface
import cv2 as cv
import sys
import tensorflow as tf
from cv2 import imread
from cv2 import CascadeClassifier
from cv2 import rectangle
from cv2 import imshow
from cv2 import waitKey
from cv2 import destroyAllWindows
from numpy import asarray
from PIL import Image
from deepface import DeepFace
 
def distance_to_prob(x):
      return 1.5*(100-x*100/0.47)
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])  


def extract_face_from_image(image_path,image_path_2, required_size=(224, 224),):
    image = plt.imread(image_path)
    detector = MTCNN()
    faces = detector.detect_faces(image)
    face_images = []
    distancearray=[]
    min_dist=0.47
    for face in faces:      
        x1, y1, width, height = face['box']
        x2, y2 = x1 + width, y1 + height
        face_boundary = image[y1:y2, x1:x2]
        face_image = Image.fromarray(face_boundary)
        face_image = face_image.resize(required_size)
        face_array = asarray(face_image)
        face_images.append(face_array)
        resp=DeepFace.verify(face_array,image_path_2,enforce_detection=False,model_name="VGG-Face") 
        if(resp["distance"]<min_dist):
          min_dist=resp["distance"]
    for face in faces:
        x1, y1, width, height = face['box']
        x2, y2 = x1 + width, y1 + height
        face_boundary = image[y1:y2, x1:x2]
        face_image = Image.fromarray(face_boundary)
        face_image = face_image.resize(required_size)
        face_array = asarray(face_image)
        face_images.append(face_array)
        resp=DeepFace.verify(face_array,image_path_2,enforce_detection=False,model_name="VGG-Face") 
        distancearray.append(resp["distance"])

        if(resp["distance"]==min_dist):        
          image=cv.rectangle(image, (x1, y1), (x2, y2), (0, 225, 0), 1)
          thickness = 2
          font=cv.FONT_HERSHEY_SIMPLEX
          fontScale = 0.45
          color = (0, 225, 0)
          image = cv.putText(image, "%"+str(int(distance_to_prob(resp["distance"]))), (x1, y1-10), font,fontScale, color, thickness, cv.LINE_AA) 
        else:    
          image=cv.rectangle(image, (x1, y1), (x2, y2), (225, 0, 0), 1)
          thickness = 2
          font=cv.FONT_HERSHEY_SIMPLEX
          fontScale = 0.5
          color = (225, 0, 0)
          image = cv.putText(image, '', (x1, y1-10), font,fontScale, color, thickness, cv.LINE_AA)     
    print(len(face_images))
    return face_images,image
def ana(pic1_path,pic2_path):
      extracted_face,sonuc = extract_face_from_image(pic1_path,pic2_path)
      plt.imshow(sonuc)
      plt.show()
      return sonuc
    
def allowed_file(filename):
  return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_form():
  return render_template('upload.html')

@app.route('/vmd_timestampback')
def vmd_timestampback():
  return render_template('upload.html')

@app.route('/vmd_timestamp')
def vmd_timestamp():
  import os
  path = os.getcwd()+"/static/outputs"   
  list_of_files = []
  for filename in os.listdir(path):
        list_of_files.append(filename)
  return render_template('astxt.html', filenamess=list_of_files)

@app.route('/', methods=['POST'])
def upload_image():
  if 'files[]' not in request.files:
    flash('No file part')
    return redirect(request.url)
  files = request.files.getlist('files[]')  
  file_names = []
  image_list=[]
  for file in files:
    if file and allowed_file(file.filename):
      filename = secure_filename(file.filename)
      print(filename)
      file_names.append(filename)
      file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
      image1 = plt.imread("static/uploads/"+filename)
      image1path="static/uploads/"+filename
      image_list.append(image1path)   
  sonuc=ana(image_list[0],image_list[1])
  img = Image.fromarray(sonuc, 'RGB')
  # Output img save outputs
  
  #sonucpath="çıktı1.jpg"
  output_file_names=file_names[0]
  sonucpath="çıktı"+output_file_names
  
  img.save("static/uploads/"+sonucpath) 
  img.save("static/outputs/"+sonucpath) 
  sonuc = Image.open(r"static/uploads/"+sonucpath) 
  file_names.append(sonucpath)
  print(file_names[0])
  print(file_names[1])
  print(file_names[2])
  new_filenames=file_names
  for i in file_names:
        del i
  return render_template('upload.html', filenames=new_filenames)

@app.route('/display/<filename>')
def display_image(filename):
  return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/display/<filename>')
def display_back_page_image(filename):
  return redirect(url_for('static', filename='outputs/' + filename), code=301)

if __name__ == "__main__":
    app.run()
 