import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime
import os
from gtts import gTTS
import pygame

# Initialize Pygame mixer for audio
pygame.mixer.init()

# Function to play audio dynamically
def speak_welcome(name):
    try:
        # Create the audio dynamically
        text = f"Welcome {name}"
        tts = gTTS(text=text, lang='en')
        audio_file = f"welcome_{name}.mp3"
        tts.save(audio_file)

        # Play the audio
        pygame.mixer.music.load(audio_file)
        pygame.mixer.music.play()

        # Delete the file after playing (optional, for cleanup)
        while pygame.mixer.music.get_busy():  # Wait until the audio finishes
            pass
        os.remove(audio_file)
    except Exception as e:
        print(f"Error in speech synthesis: {e}")

# Main Code
print("Hello")
video_capture = cv2.VideoCapture(0)

# Load Known Images
arpit_image = face_recognition.load_image_file("faces/arpit.jpg")
arpit_encoding = face_recognition.face_encodings(arpit_image)[0]

abhay_image = face_recognition.load_image_file("faces/abhay.jpg")
abhay_encoding = face_recognition.face_encodings(abhay_image)[0]

rudr_image = face_recognition.load_image_file("faces/rudr.jpg")
rudr_encoding = face_recognition.face_encodings(rudr_image)[0]

known_face_encodings = [arpit_encoding, abhay_encoding, rudr_encoding]
known_face_names = ["Arpit", "Abhay", "Rudransh"]

# List of expected students
students = known_face_names.copy()

face_locations = []
face_encodings = []

# Current date and time
now = datetime.now()
current_date = now.strftime("%d-%m-%Y")

f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Recognize faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            print(f"Recognized and Marked: {name}")

            # Display name on the screen
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, f"{name} Present", (50, 50), font, 1, (0, 255, 0), 2)

            # Mark attendance
            if name in students:
                students.remove(name)
                current_time = datetime.now().strftime("%H:%M:%S")
                lnwriter.writerow([name, current_time])

                # Speak welcome
                speak_welcome(name)

    # Show video
    cv2.imshow("Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()
