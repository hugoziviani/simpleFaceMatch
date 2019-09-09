import cv2
import dlib
import os

video_capture = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

while True:
    ret, frame = video_capture.read()
    frame = cv2.flip(frame,180)
    clahe_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = detector(clahe_image, 1)

    for k, faceItr in enumerate(faces):
        points = predictor(clahe_image, faceItr)
        print (points.parts())
        '''
        for (x, y) in points:
            print ('Ola')
        '''
        for i in range(1,68):
            cv2.circle(frame, (points.part(i).x, points.part(i).y), 1, (255,0,0), thickness=-1)
            cv2.putText(frame, str(i), (points.part(i).x,points.part(i).y),
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.3,
                    color=(0, 0, 255))
         
    cv2.imshow("Resultado", frame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

