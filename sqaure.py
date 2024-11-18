import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime
from gtts import gTTS
import pygame
pygame.mixer.init()
# Initialize video capture
video_capture = cv2.VideoCapture(0)

# Load known faces and names dynamically from the "faces" folder
known_face_encodings = []
known_face_names = []

#SPEAK
def speak_welcome(name):
    try:
        # Create the audio dynamically
        text = f"{name} has been marked present"
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

faces_path = "faces/"
for filename in os.listdir(faces_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Extract name from filename (e.g., "arpit.jpg" -> "Arpit")
        name = os.path.splitext(filename)[0].capitalize()
        image_path = os.path.join(faces_path, filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]

        known_face_encodings.append(encoding)
        known_face_names.append(name)

print(f"Loaded faces: {known_face_names}")

# Initialize attendance tracking
students = set(known_face_names)  # Use a set for efficient lookups
attended_students = set()  # Track students who have already been marked

# Prepare CSV file
now = datetime.now()
current_date = now.strftime("%d-%m-%Y")
csv_file_path = f"{current_date}.csv"
with open(csv_file_path, "w+", newline="") as f:
    lnwriter = csv.writer(f)
    lnwriter.writerow(["Name", "Time"])

# Process video frames
while True:
    _, frame = video_capture.read()

    # Resize frame for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces and encodings in the frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare faces with known encodings
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        # Identify
        name = "Unknown"
        if face_distances[best_match_index] < .38:
            if name != "Unknown":
                print(f"Recognized and Marked: {name}")

            name = known_face_names[best_match_index]

        # rectangle around the face
        top, right, bottom, left = [v * 4 for v in face_location]  # Scale back to original frame size
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # name label below face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        magenta = (255, 0, 255)  # Magenta color in BGR
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, magenta, 1)


        # Mark attendance if recognized and abseent
        if name != "Unknown" and name not in attended_students:
            attended_students.add(name)
            current_time = datetime.now().strftime("%H:%M:%S")
            with open(csv_file_path, "a", newline="") as f:
                lnwriter = csv.writer(f)
                lnwriter.writerow([name, current_time])

            # Display message on screen
            cv2.putText(frame, f"Welcome, {name}!", (50, 50), font, 1, (0, 255, 0), 2)
            print(f"Recognized and Marked: {name}")

            #Audio
            speak_welcome(name)
            print(f"Recognized and Marked: {name}")

    # video frame
    cv2.imshow("Attendance System", frame)


    # Exit on q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
pygame.mixer.quit()