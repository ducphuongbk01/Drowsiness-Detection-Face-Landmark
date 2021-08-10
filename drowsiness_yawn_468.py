
# ADD NECESSARY LIBRARY IN PROJECT

import numpy as np
import time
import cv2
import os
import mediapipe as mp
import time
from scipy.spatial import distance as dist
from threading import Thread


# Alarm to warning the driver
def alarm(msg):
    global eyes_status
    global mouth_status
    global saying

    while eyes_status:
        print('call')
        s = 'espeak "'+msg+'"'
        os.system(s)

    if mouth_status:
        print('call')
        saying = True
        s = 'espeak "' + msg + '"'
        os.system(s)
        saying = False


#Compute the EAR (EYE ASPECT RATIO)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[15])
    B = dist.euclidean(eye[2], eye[14])
    C = dist.euclidean(eye[3], eye[13])
    D = dist.euclidean(eye[4], eye[12])
    E = dist.euclidean(eye[5], eye[11])
    F = dist.euclidean(eye[6], eye[10])
    G = dist.euclidean(eye[7], eye[9])

    H = dist.euclidean(eye[0], eye[8])

    ear = (A + B + C + D + E + F + G) / (7.0 * H)

    return ear

#Compute the relation between EAR of 2 eyes
def final_ear(shape):
    leftEye = np.array([shape[33], shape[246],shape[161],shape[160],shape[159],shape[158],shape[157],shape[173],shape[133],shape[155],shape[154],shape[153],shape[145],shape[144],shape[163],shape[7]])
    rightEye = np.array([shape[263], shape[466],shape[388],shape[387],shape[386],shape[385],shape[384],shape[398],shape[362],shape[382],shape[381],shape[380],shape[374],shape[373],shape[390],shape[249]])

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return ear

#Compute the distance between top and bot of lips
def lip_distance(shape):
    top_lip = np.array([shape[37],shape[72],shape[38],shape[82],shape[0],shape[11],shape[12],shape[13],shape[267],shape[302],shape[268],shape[312]])

    low_lip = np.array([shape[84],shape[85],shape[86],shape[87],shape[14],shape[15],shape[16],shape[17],shape[317],shape[316],shape[315],shape[314]])

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance

#The class which use to detect the 468 points face landmark
class FaceMeshDetector():
    def __init__(self, staticMode = False, maxFaces = 1, minDetectionCon = 0.5, minTrackCon = 0.5):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(self.staticMode, self.maxFaces, self.minDetectionCon, self.minTrackCon)
        self.drawSpec = self.mpDraw.DrawingSpec(thickness = 1, circle_radius = 1)

    def findFaceMesh(self, img, draw = True):
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for faceLms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACE_CONNECTIONS, self.drawSpec, self.drawSpec)
                
                face = []
                for id, lm in enumerate(faceLms.landmark):
                    ih, iw, ic = img.shape
                    x,y = int(lm.x*iw), int(lm.y*ih)
                    face.append([x, y])
                face = np.array(face)
                faces.append(face)
        return img, faces

#Declare variable
EYE_AR_THRESH = 0.18
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 25
eyes_status = False
mouth_status = False
saying = False
COUNTER = 0


#MAIN function
def main():
    global EYE_AR_THRESH
    global EYE_AR_CONSEC_FRAMES
    global YAWN_THRESH
    global eyes_status
    global mouth_status
    global saying
    global COUNTER
    pTime = 0
    print("-> Loading the predictor and detector...")
    detector = FaceMeshDetector() #Call class detection
    print("-> Starting Video Stream")
    cap = cv2.VideoCapture(0)
    time.sleep(1.0) #Timer for system to repair all thing
    while True:
        ret, img = cap.read()
        if not ret:
            continue
        try: 
            img, faces = detector.findFaceMesh(img, draw = True)
            ear = final_ear(faces[0])
            distance = lip_distance(faces[0])
            
            if ear < EYE_AR_THRESH:
                COUNTER += 1

                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    if eyes_status == False:
                        eyes_status = True
                        t = Thread(target=alarm, args=('wake up, wake up, wake up sir',))
                        t.deamon = True
                        t.start()

                    cv2.putText(img, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER = 0
                eyes_status = False

            if (distance > YAWN_THRESH):
                if mouth_status == False and saying == False:
                    mouth_status = True
                    t = Thread(target=alarm, args=('take some fresh air sir',))
                    t.deamon = True
                    t.start()
                cv2.putText(img, "YAWN ALERT!", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                mouth_status = False

            cv2.putText(img, "EAR: {:.2f}".format(ear), (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(img, "YAWN: {:.2f}".format(distance), (500, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        except:
            cv2.putText(img,"Cannot find the face!", (10, 30), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)            
        
        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (10, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        cv2.imshow("Video", img)

        

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()


