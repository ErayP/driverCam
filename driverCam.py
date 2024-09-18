import cv2
import mediapipe as mp
import time

# Göz indeksleri
LEFT_EYE_INDEXES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDEXES = [33, 160, 158, 133, 153, 144]

nose_p = 4


# Göz kırpma oranı hesaplama
def eye_aspect_ratio(eye_landmarks, frame):
    horizontal_length = ((eye_landmarks[0][0] - eye_landmarks[3][0]) ** 2 + (
                eye_landmarks[0][1] - eye_landmarks[3][1]) ** 2) ** 0.5
    vertical_length = ((eye_landmarks[1][0] - eye_landmarks[5][0]) ** 2 + (
                eye_landmarks[1][1] - eye_landmarks[5][1]) ** 2) ** 0.5
    ratio = vertical_length / horizontal_length
    return ratio


# Landmark'ları çıkarma
def extract_eye_landmarks(landmarks, indexes):
    return [(landmarks[idx].x, landmarks[idx].y) for idx in indexes]


def extract_nose_landmarks(landmarks, index):
    return [landmarks[index].x, landmarks[index].y]


def faceFocus(landmark, h, w):
    x, y = landmark[0] * w, landmark[1] * h
    if x < 200 or x > 440 or y < 150 or y > 330:
        return True
    else:
        return False


# MediaPipe ayarları
mp_face_mesh = mp.solutions.face_mesh
mp_face_draw = mp.solutions.drawing_utils

blink_count = 0
closed_eye_frame = 0
eye_closed_time = 0.0
eye_closed_start_time = None
total_eye_close_count = 0

start_time = None  # Yüz odaktan çıktığında zaman tutmak için

with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5,
                           min_tracking_confidence=0.5) as face_mesh:
    # Kamera başlatma
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Kamera görüntüsü alınamadı.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Sol ve sağ göz landmark'larını çıkar
                left_eye_landmarks = extract_eye_landmarks(face_landmarks.landmark, LEFT_EYE_INDEXES)
                right_eye_landmarks = extract_eye_landmarks(face_landmarks.landmark, RIGHT_EYE_INDEXES)

                # Göz kırpma oranlarını hesapla
                left_eye_ratio = eye_aspect_ratio(left_eye_landmarks, frame)
                right_eye_ratio = eye_aspect_ratio(right_eye_landmarks, frame)

                # Gözlerin kapalı olup olmadığını kontrol et
                if left_eye_ratio < 0.2 and right_eye_ratio < 0.2:
                    closed_eye_frame += 1
                    if eye_closed_start_time is None:
                        eye_closed_start_time = time.time() - eye_closed_time

                else:
                    h, w, _ = frame.shape
                    nose_mark = extract_nose_landmarks(face_landmarks.landmark, nose_p)
                    boolean = faceFocus(nose_mark, h, w)

                    if boolean:
                        if start_time is None:
                            start_time = time.time()  # İlk kez odak kaybolursa zamanı başlat

                        elif time.time() - start_time >= 4:
                            cv2.putText(frame, "Lutfen yola odaklaniniz.",(10, 470),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 255, 0), 2)
                            start_time = time.time()  # 4 saniyeyi tekrar başlatmak için sıfırla

                    else:
                        start_time = None  # Odak tekrar sağlandığında sıfırla
                        focus_warning_displayed = False  # Uyarıyı tekrar gösterebiliriz

                    if closed_eye_frame > 1:
                        blink_count += 1
                        closed_eye_frame = 0

                        if eye_closed_start_time is not None:
                            eye_closed_time = time.time() - eye_closed_start_time
                            eye_closed_start_time = None

                            if eye_closed_time > 0.5:
                                total_eye_close_count += 1
                            eye_closed_time = 0

                            if total_eye_close_count > 3:
                                cv2.putText(frame, "Uykulusunuz lutfen dinlenin.", (10, 470),cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 255, 0), 2)
                                total_eye_close_count = 0

        # Göz kırpma sayısını göster
        #cv2.line(frame, pt1=(200, 0), pt2=(200, 480), color=(0, 255, 255), thickness=2)
        #cv2.line(frame,pt1=(440,0),pt2=(440,480),color=(0,255,255),thickness=2)
        #cv2.line(frame,pt1=(0,150),pt2=(640,150),color=(0,255,255),thickness=2)
        #cv2.line(frame,pt1=(0,330),pt2=(640,330),color=(0,255,255),thickness=2)
        cv2.putText(frame, "Goz Kirpma Sayisi: {}".format(blink_count), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)
        cv2.imshow('Arac kamerasi', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
