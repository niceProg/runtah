from cv2 import VideoCapture
import cv2
from flask import Flask, render_template, request, Response
import face_recognition
import numpy as np
# from absen import Recognition
# import file
import os

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/absensi')
def absensi():
    return render_template('/absensi.html')

@app.route('/profil')
def profil():
    return render_template('/profil.html')

@app.route('/help')
def help():
    return render_template('/help.html')

camera = cv2.VideoCapture(0)

image_1 = face_recognition.load_image_file("Img_Tester/nizar/nz (3).jpg")
image_1_face_encoding = face_recognition.face_encodings(image_1)[0]

image_2 = face_recognition.load_image_file("Img_Tester/annon/an (5).jpg")
image_2_face_encoding = face_recognition.face_encodings(image_2)[0]
        
image_3 = face_recognition.load_image_file("Img_Tester/khaepah/kh (4).jpg")
image_3_face_encoding = face_recognition.face_encodings(image_3)[0]

image_4 = face_recognition.load_image_file("Img_Tester/agung/ag (3).jpg")
image_4_face_encoding = face_recognition.face_encodings(image_4)[0]

image_5 = face_recognition.load_image_file("Img_Tester/bayu/by (3).jpg")
image_5_face_encoding = face_recognition.face_encodings(image_5)[0]

image_6 = face_recognition.load_image_file("Img_Tester/gusti/gs (3).jpg")
image_6_face_encoding = face_recognition.face_encodings(image_6)[0]

image_7 = face_recognition.load_image_file("Img_Tester/krismawati/krs (6).jpg")
image_7_face_encoding = face_recognition.face_encodings(image_7)[0]

image_8 = face_recognition.load_image_file("Img_Tester/prana/pr (3).jpg")
image_8_face_encoding = face_recognition.face_encodings(image_8)[0]

image_9 = face_recognition.load_image_file("Img_Tester/ramdon/rm (3).jpg")
image_9_face_encoding = face_recognition.face_encodings(image_9)[0]

image_10 = face_recognition.load_image_file("Img_Tester/yumna/ym (50).jpg")
image_10_face_encoding = face_recognition.face_encodings(image_10)[0]

# image_11 = face_recognition.load_image_file("Img_Tester/lia/lia (9).jpg")
# image_11_face_encoding = face_recognition.face_encodings(image_11)[0]

# image_12 = face_recognition.load_image_file("Img_Tester/gina/gn (9).jpg")
# image_12_face_encoding = face_recognition.face_encodings(image_12)[0]

# image_12 = face_recognition.load_image_file("Img_Tester/nirla/nr (9).jpg")
# image_12_face_encoding = face_recognition.face_encodings(image_12)[0]

# image_13 = face_recognition.load_image_file("Img_Tester/novita/nv (9).jpg")
# image_13_face_encoding = face_recognition.face_encodings(image_13)[0]

# image_14 = face_recognition.load_image_file("Img_Tester/nurul/nrl (9).jpg")
# image_14_face_encoding = face_recognition.face_encodings(image_14)[0]

# image_15 = face_recognition.load_image_file("Img_Tester/risma/rsm (9).jpg")
# image_15_face_encoding = face_recognition.face_encodings(image_15)[0]

# image_16 = face_recognition.load_image_file("Img_Tester/rissa/rs (9).jpg")
# image_16_face_encoding = face_recognition.face_encodings(image_16)[0]


# Create arrays of known face encodings and their names
known_face_encodings = [
    image_1_face_encoding,
    image_2_face_encoding,
    image_3_face_encoding,
    image_4_face_encoding,
    image_5_face_encoding,
    image_6_face_encoding,
    image_7_face_encoding,
    image_8_face_encoding,
    image_9_face_encoding,
    image_10_face_encoding
    # image_11_face_encoding
    # image_12_face_encoding
    # image_12_face_encoding
    # image_13_face_encoding
    # image_14_face_encoding,
    # image_15_face_encoding,
    # image_16_face_encoding
]

known_face_names = [
    "Miftakhul Nizar",
    "Annon Pri Antomo",
    "Khaepah",
    "Agung Iswanto",
    "Saksono Bayu Aji",
    "Gusti Robbani",
    "Krismawati",
    "Pranaditya",
    "Ramdon Baekhaqi",
    "Yumna"
    # "Rizki Amalia"
    # "Gina Sonia Wiranti"
    # "Nirla Wahidatussalam"
    # "Novita Fitria Putri"
    # "Siti Nurul Ulumi",
    # "Risma Nian Kupandang",
    # "Aulia Rizki Syarifah"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
           
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Tidak Diketahui"
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                print(matches,face_distances)
                
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)
            

            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left,bottom-35), (right,bottom), (0,0,255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame,name, (left+6,bottom-6), font,1, (255,255,255), 2)
                # Akurasi
                cv2.putText(frame,f' {round(face_distances[0],2)}',(50,50),font,1,(0,0,255),2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(debug=True)