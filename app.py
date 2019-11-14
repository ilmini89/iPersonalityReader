#Usage: python app.py
import os
from flask import Flask, render_template, request, redirect, url_for
from werkzeug import secure_filename
from keras.preprocessing.image import image ,img_to_array
import numpy as np
import uuid
from keras.backend import clear_session
from PIL import Image
import requests
from keras.models import load_model
from keras import applications
from keras.applications.mobilenet_v2 import preprocess_input
import time
import cv2





img_width, img_height = 224, 224
target_size=224,224
model_path = './models/trained_model.model'

#model = load_model(model_path)
#model.load_weights(model_weights_path)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg'])

def get_as_base64(url):
    return base64.b64encode(requests.get(url).content)

def predictx(file,filename):
    print("XXXXXXXXXXXXXXXXXXXXXX")
    
    CASCADE="./res/Face_cascade.xml"
    FACE_CASCADE=cv2.CascadeClassifier(CASCADE)
    image1=cv2.imread(file)
    image_grey=cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
    faces = FACE_CASCADE.detectMultiScale(image_grey,scaleFactor=1.16,minNeighbors=5,minSize=(100,100),flags=0)
    print(faces)
    
    filename2="original_"+ filename
    file_path1 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
    print(file_path1)
    cv2.imwrite(file_path1,image1)
    for x,y,w,h in faces:
        
        sub_img=image1[y-10:y+h+10,x-10:x+w+10]
        
        cv2.imwrite(file,sub_img)
        
            #os.chdir("../")
            #cv2.rectangle(image,(x,y),(x+w,y+h),(255, 255,0),2)
            #cv2.imshow("Faces Found",image)
    
    
       
    img = Image.open(file)
    
    if img.size != target_size:
        img = img.resize(target_size)
    
    print("ok1")
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    print("ok2")
    x = preprocess_input(x)
    print("ok3")
    clear_session()
    model = load_model(model_path)
    array = model.predict(x)
    print("ok4")
    print(array)
    print("ok5")
    
    return array


def my_random_string(string_length=10):
    """Returns a random string of length string_length."""
    random = str(uuid.uuid4()) # Convert UUID format to a Python string.
    random = random.upper() # Make all characters uppercase.
    random = random.replace("-","") # Remove the UUID '-'.
    return random[0:string_length] # Return the random string.

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



@app.route("/")
def template_test():
    #model = load_model(model_path)
    return render_template('template.html', extraversion = '', neuroticism = '', agreeableness = '', conscientiousness = '', interview = '', openness='', imagesource='../uploads/template.jpeg', imagesource2='../uploads/template.jpeg')

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        import time
        start_time = time.time()
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            
            
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            print(filename)
            file.save(file_path)
            result = predictx(file_path,filename)
            filenamex=filename
            '''if result == 0:
                label = 'Daisy'
            elif result == 1:
                label = 'Rose'			
            elif result == 2:
                label = 'Sunflowers'''
            print(result[0][0][0])
            print(result[1][0][0])
            print(result[2][0][0])
            print(result[3][0][0])
            print(result[4][0][0])
            print(result[5][0][0])
            print(file_path)
            filename = my_random_string(6) + filename

            os.rename(file_path, os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print("--- %s seconds ---" % str (time.time() - start_time))
            return render_template('template.html', extraversion = round(result[0][0][0],4), neuroticism = round(result[1][0][0],4), agreeableness = round(result[2][0][0],4), conscientiousness = round(result[3][0][0],4), interview = round(result[4][0][0],4), openness=round(result[5][0][0],4),imagesource='../uploads/' + filename, imagesource2='../uploads/original_'+filenamex )

from flask import send_from_directory

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

from werkzeug import SharedDataMiddleware
app.add_url_rule('/uploads/<filename>', 'uploaded_file',
                 build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads':  app.config['UPLOAD_FOLDER']
})





if __name__ == "__main__":
    
    app.run()