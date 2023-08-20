import asyncio # imports required libraries
from websockets.server import serve
import base64
import cv2
import numpy as np
import mediapipe as mp
import math
from numpy import expand_dims
import h5py
from keras.models import load_model, model_from_json
import mediapipe as mp
import tensorflow.compat.v1 as tf
import pickle
tf.disable_v2_behavior()

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
model = load_model('facenet_keras.h5')
model.load_weights('facenet_keras_weights.h5')
camw = 640
camh = 480

uuidembeddings = {}

# For webcam input:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) 
def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
def get_embedding(model, face_pixels1):
    try:
        face_pixels = cv2.resize(face_pixels1, (160,160)).astype('float32')
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        samples = expand_dims(face_pixels, axis=0)
        yhat = model.predict(samples)
        return yhat[0]    
    except:
        print("whoops")
        return "noix deez"
async def sockanal(websocket): # Analyzes websocket data
    global numframes
    global ipkeys
    global roomcrowds

    async for fmessage in websocket:       
        userid = fmessage[:36] # gets header from websocket data
        message = fmessage[36:]
        if message[:100] == "data:,":
            await websocket.send("owo")
        else:
            try: 
                thisface = uuidembeddings[userid][4]
                await websocket.send("Facial data extraction complete")            
            except:

                nparr = np.frombuffer(base64.b64decode(message.split(',')[1]), np.uint8) # decodes base64 image to Numpy matrix
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # converts matrix to opencv format image

                image = cv2.resize(img, (640, 480))
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_mesh.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                silhouette = [
                10,  338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
                397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
                172, 58,  132, 93,  234, 127, 162, 21,  54,  103, 67,  109
                ]
                newimage = image
                face = image
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:        
                        xcoords = [handmark.x for handmark in face_landmarks.landmark]
                        ycoords = [handmark.y for handmark in face_landmarks.landmark]
                        sxcoords = [int(xcoords[i] * camw) for i in silhouette]
                        sycoords = [int(ycoords[i] * camh) for i in silhouette]
                        rex = int(xcoords[33] * camw)
                        rey = int(ycoords[33] * camh)           
                        lex = int(xcoords[263] * camw)
                        ley = int(ycoords[263] * camh)
                        midx = int((rex + lex)/2)
                        midy = int((rey + ley)/2)
                        angle = math.atan((rey-ley)/(rex-lex))
                        deg = int(360 + (angle * (180/3.14159))) % 360
                        M = cv2.getRotationMatrix2D((midx, midy), deg, 1.0)
                        sxx = [int(M[0][0] * sxcoords[i] + M[0][1] * sycoords[i] + M[0][2]) for i in range(len(sycoords))]
                        sxy = [int(M[1][0] * sxcoords[i] + M[1][1] * sycoords[i] + M[1][2]) for i in range(len(sycoords))]
                        newimage = cv2.warpAffine(image, M, (640, 480))            
                        minx = min(sxx)
                        maxx = max(sxx)
                        miny = min(sxy)
                        maxy = max(sxy)
                        face = newimage[miny:maxy, minx:maxx]
                        cv2.rectangle(newimage, (minx, miny), (maxx, maxy), (0,0,255), 3) 
                                                
                        facevector = get_embedding(model, face)
                        if not facevector == "noix deez":
                            thislist = []
                            try:
                                thislist = uuidembeddings[userid]
                            except:
                                thislist = []
                                uuidembeddings[userid] = thislist
                            toadd = True
                            for i in thislist:
                                if findCosineSimilarity(i, facevector) <= 0.2:
                                    toadd = False
                                
                            if toadd:
                                print("NEW FACE ADDED")
                                thislist.append(facevector)
                            else:
                                print("EXISTING FACE")
                            uuidembeddings[userid] = thislist
                            if len(thislist) == 5:
                                f = open(f'{userid}.csfd', 'wb')
                                pickle.dump(thislist, f)
                                f.close()         
                                print("finished")                                                                                 
                await websocket.send("Awaiting next face image") # returns success message


async def main():
    async with serve(sockanal, "0.0.0.0", 5570): # serves socket with input sent to the analysis function
        print("the websocket server has started")
        await asyncio.Future()  # run forever

asyncio.run(main()) # runs main function asynchronously

if DISPLAY: # closes all video stream windows on program close
    cv2.destroyAllWindows()