import cv2
import os
import mediapipe as mp
import numpy as np


def create_dataset():
    cap = cv2.VideoCapture(0)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    output_dir = 'dataset'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            count += 1
            face = frame[y:y + h, x:x + w]
            face = cv2.resize(face, (200, 200))
            cv2.imwrite(os.path.join(output_dir, f'user_{count}.jpg'), face)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('Face Dataset Collection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 100:
            break

    cap.release()
    cv2.destroyAllWindows()


def train_face_recognizer(data_dir):
    faces = []
    labels = []

    label_map = {}
    current_label = 0

    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('jpg'):
                path = os.path.join(root, file)
                label = os.path.basename(root)
                if label not in label_map:
                    label_map[label] = current_label
                    current_label += 1
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                faces.append(img)
                labels.append(label_map[label])

    faces = np.array(faces)
    labels = np.array(labels)

    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces, labels)

    return face_recognizer, label_map

def count_fingers(results):
    finger_count = 0
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            finger_count = 0
            for id, lm in enumerate(hand_landmarks.landmark):
                if id in [8, 12, 16, 20]:
                    if lm.y < hand_landmarks.landmark[id - 2].y:
                        finger_count += 1
    return finger_count


def realtime(face_recognizer, label_map):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2)
    mp_draw = mp.solutions.drawing_utils

    # face_recognizer, label_map = train_face_recognizer('dataset')
    label_map_reversed = {v: k for k, v in label_map.items()}

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face = gray[y:y + h, x:x + w]
            face = cv2.resize(face, (200, 200))

            label, confidence = face_recognizer.predict(face)
            name = label_map_reversed[label] if confidence < 60 else "Unknown"
            print(name, confidence)
            if name == "dataset":
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(img_rgb)

                num_fingers = count_fingers(results)

                if num_fingers == 1:
                    cv2.putText(frame, "Marina", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                elif num_fingers == 2:
                    cv2.putText(frame, "Labenskaya", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            else:
                cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # create_dataset()
    face_recognizer, label_map = train_face_recognizer('dataset')
    realtime(face_recognizer, label_map)
