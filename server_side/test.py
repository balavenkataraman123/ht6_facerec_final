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
import os
tf.disable_v2_behavior()


UUID_OVERRIDE = "af521dcd-eff4-427e-ba77-b84b3722d50b"

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
model = load_model('facenet_keras.h5')
model.load_weights('facenet_keras_weights.h5')
camw = 640
camh = 480

uuidembeddings = {}
res = []
uidtemps = {}

for file in os.listdir("."):
    if file.endswith('.csfd'):
        res.append(file)
for fname in res:
    f = open(fname, 'rb')
    uuidembeddings[fname[:-5]] = []
    while True:
        try:
           for d in pickle.load(f):
                uuidembeddings[fname[:-5]].append(d)
        except:
            break
print(uuidembeddings)

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
        return "no"
async def sockanal(websocket): # Analyzes websocket data
    global numframes
    global ipkeys
    global roomcrowds

    async for fmessage in websocket:               
        userid = fmessage[:36] # gets header from websocket data
        if UUID_OVERRIDE != "no":
            userid = UUID_OVERRIDE
        message = fmessage[36:]
        if message[:100] == "data:,":
            await websocket.send("owo")
        else:
            print(fmessage[:100])
            print(message[:100])
            print(userid)
            l = []
            try:
                l = uidtemps[userid]
            except:
                uidtemps[userid] = []
            nparr = np.frombuffer(base64.b64decode(message.split(',')[1]), np.uint8)# decodes base64 image to Numpy matrix
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # converts matrix to opencv format image
            image = cv2.resize(img, (640, 480))
            cv2.imshow('MediaPipe Face Mesh2', image)    

            if cv2.waitKey(5) & 0xFF == 27:
                break            
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
            cs = 20              

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
                    if facevector != "no":
                        for i in uuidembeddings[userid]:
                            cs = min(cs, findCosineSimilarity(facevector, i))
            print(cs)
            l.append(cs)
            succ = 0
            if len(l) >= 10:
                overthreshold = 0
                for k in l[-10:]:
                    if k > 0.28:
                        overthreshold += 1
                if overthreshold <= 3:
                    print("FACE MATCH DETECTEDDDD")
                    succ = 1                    
                    await websocket.send("FACE MATCH SUCCESS")
                else:
                    l.pop(0)
                    
                    
            #if len(l) >= 100 and not not succ:
            #    await websocket.send("FACE MATCH FAILED")
            #elif not succ:
            await websocket.send("AWAITING IMAGE DATA") # returns success message


async def main():
    async with serve(sockanal, "0.0.0.0", 8765): # serves socket with input sent to the analysis function
        print("the websocket server has started")
        await asyncio.Future()  # run forever

asyncio.run(main()) # runs main function asynchronously

