#modified by Hugo Ziviani
#This code was generated based on the reference: https://pypi.org/project/face_recognition/
import face_recognition
import cv2
import numpy as np

videoCapture = cv2.VideoCapture(0)

faceH = face_recognition.load_image_file("faces/hugo.jpg")
faceR = face_recognition.load_image_file("faces/Rafael.png")
faceB = face_recognition.load_image_file("faces/Burle.png")
faceD = face_recognition.load_image_file("faces/Denise.png")

srcFaceH = face_recognition.face_encodings(faceH)[0]
srcFaceR = face_recognition.face_encodings(faceR)[0]
srcFaceB = face_recognition.face_encodings(faceB)[0]
srcFaceD = face_recognition.face_encodings(faceD)[0]
#print (faceEncode)

knownFaces = [srcFaceH, srcFaceR, srcFaceB, srcFaceD]
knownNames = ["Hugo", "Rafael", "Burle", "Denise"]

while True:
    r, bgrFrame = videoCapture.read()
    rgbFrame = cv2.cvtColor(bgrFrame, cv2.COLOR_BGR2RGB)

    facePoints = face_recognition.face_locations(rgbFrame)
    #print (facePoints)
    foundFaces = face_recognition.face_encodings(rgbFrame, facePoints)
    #print (facesEncodings)
    
    for (top, right, bottom, left), foundFaces in zip(facePoints, foundFaces):
        nameToDisplay = "Nao sei..."
        cases = face_recognition.compare_faces(knownFaces, foundFaces)
        #print (cases)
        idCaseFoud = face_recognition.face_distance(knownFaces, foundFaces)
        indexCase = np.argmin(idCaseFoud)
        if cases[indexCase]:
            nameToDisplay = knownNames[indexCase]
        cv2.rectangle(bgrFrame, (left, top), (right, bottom), (255, 255, 0), 2)
        cv2.putText(bgrFrame, nameToDisplay, (left + 8, bottom - 8), 3, 1.0, (0, 0, 255), 1)

    cv2.imshow("Resultado-BGR", bgrFrame)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        break

videoCapture.release()
cv2.destroyAllWindows()
