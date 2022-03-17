from keras.models import Sequential, Model
from keras.layers import Flatten, Dropout, Activation, Permute
from keras.layers import Convolution2D, MaxPooling2D
from keras import backend as K
K.set_image_data_format( 'channels_last' )
import numpy as np
import cv2
from scipy.spatial.distance import cosine as dcos
from scipy.io import loadmat
import os
from multiprocessing.dummy import Pool
from threading import Thread
from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image

# Fonction permettant de détecter des visages dans le champ de vision de la caméra
# Et de délimiter les contours du visage
def auto_crop_image(image):    
    if image is not None:
        im = image.copy()
        #Chargement de HaarCascade a partir du fichier avec OpenCV
        faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        
        # Lecture de l'image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(gray, 1.2, 5)

        if len(faces) > 0:
            # Dessine un rectangle autour des visages
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)        
            (x, y, w, h) = faces[0]
            center_x = x+w/2
            center_y = y+h/2
            height, width, channels = im.shape
            b_dim = min(max(w,h)*1.2,width, height)
            box = [center_x-b_dim/2, center_y-b_dim/2, center_x+b_dim/2, center_y+b_dim/2]
            box = [int(x) for x in box]
            # Rognage de l'image
            if box[0] >= 0 and box[1] >= 0 and box[2] <= width and box[3] <= height:
                crpim = im[box[1]:box[3],box[0]:box[2]]
                crpim = cv2.resize(crpim, (224,224), interpolation = cv2.INTER_AREA)
                #print("Found {0} faces!".format(len(faces)))
                return crpim, image, (x, y, w, h)
    return None, image, (0,0,0,0)

# Chargement d'un reseau de neurones convolutif pour générer des vecteurs
# A partir des visages
def convblock(cdim, nb, bits=3):
    L = []
    for k in range(1,bits+1):
        convname = 'conv'+str(nb)+'_'+str(k)
        L.append( Convolution2D(cdim, kernel_size=(3, 3), padding='same', activation='relu', name=convname) )
    L.append( MaxPooling2D((2, 2), strides=(2, 2)) )
    return L

# Reformation du modèle
def vgg_face_blank():
    withDO = True
    if True:
        mdl = Sequential()
        mdl.add( Permute((1,2,3), input_shape=(224,224,3)) )
        for l in convblock(64, 1, bits=2):
            mdl.add(l)
        for l in convblock(128, 2, bits=2):
            mdl.add(l)        
        for l in convblock(256, 3, bits=3):
            mdl.add(l)            
        for l in convblock(512, 4, bits=3):
            mdl.add(l)            
        for l in convblock(512, 5, bits=3):
            mdl.add(l)        
        mdl.add( Convolution2D(4096, kernel_size=(7, 7), activation='relu', name='fc6') )
        if withDO:
            mdl.add( Dropout(0.5) )
        mdl.add( Convolution2D(4096, kernel_size=(1, 1), activation='relu', name='fc7') )
        if withDO:
            mdl.add( Dropout(0.5) )
        mdl.add( Convolution2D(2622, kernel_size=(1, 1), activation='relu', name='fc8') )
        mdl.add( Flatten() )
        mdl.add( Activation('softmax') )
        
        return mdl
    
    else:
        raise ValueError('not implemented')
        
def copy_mat_to_keras(kmodel, l):
    kerasnames = [lr.name for lr in kmodel.layers]
    prmt = (0,1,2,3)

    for i in range(l.shape[1]):
        matname = l[0,i][0,0].name[0]
        if matname in kerasnames:
            kindex = kerasnames.index(matname)
            l_weights = l[0,i][0,0].weights[0,0]
            l_bias = l[0,i][0,0].weights[0,1]
            f_l_weights = l_weights.transpose(prmt)
            assert (f_l_weights.shape == kmodel.layers[kindex].get_weights()[0].shape)
            assert (l_bias.shape[1] == 1)
            assert (l_bias[:,0].shape == kmodel.layers[kindex].get_weights()[1].shape)
            assert (len(kmodel.layers[kindex].get_weights()) == 2)
            kmodel.layers[kindex].set_weights([f_l_weights, l_bias[:,0]])

# Recherche du vecteur le plus proche dans la base de donnees
def generate_database(featuremodel, folder_img = "photos"):
    database = {}
    for the_file in os.listdir(folder_img):
        file_path = os.path.join(folder_img, the_file)
        try:
            if os.path.isfile(file_path):
               name = the_file.split(".")[0]
               img = cv2.imread(file_path)
               crpim, srcimg, (x, y, w, h) = auto_crop_image(img)
               vector_image = crpim[None,...]
               database[name] = featuremodel.predict(vector_image)[0,:]
        except Exception as e:
            print(e)
    return database

def find_closest(featuremodel, img, database, min_detection=2.5):
    imarr1 = np.asarray(img)
    imarr1 = imarr1[None,...]

    #Prediction
    fvec1 = featuremodel.predict(imarr1)[0,:]
    print(fvec1)
    #Personne la plus proche dans la base de données
    dmin = 0.0
    umin = ""
    for key, value in database.items():
        fvec2 = value
        dcos_1_2 = dcos(fvec1, fvec2)
        if umin == "":
            dmin = dcos_1_2
            umin = key
        elif dcos_1_2 < dmin:
            dmin = dcos_1_2
            umin = key
    if dmin > min_detection:
        umin = ""
    if dmin > 0.31:
        umin = "Inconnu"
    print(umin, dmin)
    return umin, dmin

# Fonction estimant l'age de la personne
def detect_age(cap): 
    ageProto="age_deploy.prototxt"
    ageModel="age_net.caffemodel"
    
    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"
    
    classifier =load_model('./Emotion_Detection.h5')

    MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
    ageList=['(0-10)', '(12-14)', '(16-20)', '(21-25)', '(26-35)', '(38-43)', '(48-53)', '(60-80)']
    ageNet=cv2.dnn.readNet(ageModel,ageProto)
    
    genderList=['Homme','Femme']
    genderNet = cv2.dnn.readNet(genderModel, genderProto)
    
    class_labels = ['Angry','Happy','Neutral','Sad','Surprise']
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    
    while True:
        # Grab a single frame of video
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
    
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h,x:x+w]
            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
    
    
            if np.sum([roi_gray])!=0:
                roi = roi_gray.astype('float')/255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi,axis=0)
    
                blob = cv2.dnn.blobFromImage(frame, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
                
                ageNet.setInput(blob)
                agePreds = ageNet.forward()
                age = ageList[agePreds[0].argmax()]
                
                genderNet.setInput(blob)
                genderPreds = genderNet.forward()
                gender = genderList[genderPreds[0].argmax()]
                
                preds = classifier.predict(roi)[0]
                print("\nprediction = ",preds)
                label=class_labels[preds.argmax()]
                return age,gender,label


def webcam_face_recognizer(database, featuremodel):
    vc = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
    path = os.getcwd()
    videoWriter = cv2.VideoWriter(path + '\\test.avi', fourcc, 30.0, (640,480))
    vc.set(cv2.CAP_PROP_BUFFERSIZE, 3)
    ready_to_detect_identity = True
    name = ""
    cpt = 0
    senti = ""
    gender=""
    age=""
    while vc.isOpened():
        cpt += 1
        _, frame = vc.read()
        
        if frame is not None:
            im = cv2.blur(frame,(15,15))

            faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            
            faces = faceCascade.detectMultiScale(gray, 1.2, 5)
            
            grays = []
            if len(faces) > 0:
                # Dessin d'un rectange autour des visages trouves
                for i in range(0, len(faces)):
                    (x, y, w, h) = faces[i]
                    grays.append(gray[y:y+w, x:x+h])
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    center_x = x+w/2
                    center_y = y+h/2
                    height, width, channels = im.shape
                    b_dim = min(max(w,h)*1.2,width, height)
                    box = [center_x-b_dim/2, center_y-b_dim/2, center_x+b_dim/2, center_y+b_dim/2]
                    box = [int(x) for x in box]
                    # Crop Image
                    if box[0] >= 0 and box[1] >= 0 and box[2] <= width and box[3] <= height:
                        imgcrop = im[box[1]:box[3],box[0]:box[2]]
                        imgcrop = cv2.resize(imgcrop, (224,224), interpolation = cv2.INTER_AREA)

                    else:
                        imgcrop = None
                        (x, y, w, h) = (0,0,0,0)
                    
                    # Analyse de l'image
                    if ready_to_detect_identity and imgcrop is not None:
                        
                        ready_to_detect_identity = False
                        pool = Pool(processes=2)
                        name, ready_to_detect_identity = pool.apply_async(recognize_image, [featuremodel, imgcrop, database]).get()

            
                        age,gender,senti=detect_age(vc)
                        pool.close()
                    cv2.putText(img = frame, text = name+ " : ", org = (int(x),int(y+h+20)), fontFace = cv2.FONT_HERSHEY_SIMPLEX, thickness= 2, fontScale = 0.7, color = (0, 255, 0))
                    test=[gender,age,senti]
                    cnt=20
                    for i in test:
                        cnt=cnt+20
                        cv2.putText(img = frame, text = i, org = (int(x),int(y+h+cnt)), fontFace = cv2.FONT_HERSHEY_SIMPLEX, thickness= 2, fontScale = 0.7, color = (0, 255, 0))
                    cv2.imshow("Preview", frame)
                    videoWriter.write(frame) 
        if cv2.waitKey(1) == 27: # Quitter l'appli en appuyant sur echap
            break
    vc.release()
    videoWriter.release()
    cv2.destroyAllWindows()
    

    

# Fonction recuperant le nom et l'image de la personne avec le visage
# Le plus proche du visage detecte
def recognize_image(featuremodel, img, database):
    name, dmin = find_closest(featuremodel, img ,database)
    return name, True


def launch_algo():
    # Initialisation du modèle de réseau de neurones convolutif
    facemodel = vgg_face_blank()
    # Load the pretrained weights into the model
    # Chargement des poinds pré-entrainés dans le modèle
    data = loadmat('vgg-face.mat', matlab_compatible=False, struct_as_record=False)
    l = data['layers']
    description = data['meta'][0,0].classes[0,0].description

    copy_mat_to_keras(facemodel,l)
    # Final model that can get inputs and generate a prediction as an output
    # Modèle final qui peut obtenir les entrées et generer une prediction en tant que sortie
    featuremodel = Model(inputs = facemodel.layers[0].input, outputs = facemodel.layers[-2].output )
    
    db = generate_database(featuremodel)
    
    webcam_face_recognizer(db, featuremodel)
