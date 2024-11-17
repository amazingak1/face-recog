#ADD ONLY PHOTO IN /FACES , IT WILL READ NAME FROM IT  :)
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

        # Delete the file after playing  cleanup
        while pygame.mixer.music.get_busy():  # Wait until the audio finishes
            pass
        os.remove(audio_file)
    except Exception as e:
        print(f"Error in speech synthesis: {e}")

# Load all images dynamically from the "faces" folder
def load_known_faces(directory="faces/"):
    known_face_encodings = []
    known_face_names = []

    # Scan the directory for image files
    for file_name in os.listdir(directory):
        if file_name.endswith(".jpg") or file_name.endswith(".png"):  # Supported file formats
            name = os.path.splitext(file_name)[0]  # Extract the name from the file name
            image_path = os.path.join(directory, file_name)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)
            if encoding:  # Ensure encoding exists
                known_face_encodings.append(encoding[0])
                known_face_names.append(name)

    return known_face_encodings, known_face_names

# Main Program
print("Initializing System...")
video_capture = cv2.VideoCapture(0)

# Load known faces
known_face_encodings, known_face_names = load_known_faces()

# List of students
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
