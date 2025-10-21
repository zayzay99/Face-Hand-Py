import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Tidak bisa membuka kamera.")
    exit()
tip_ids = [4, 8, 12, 16, 20]

print("Memulai deteksi tangan... Tekan 'q' untuk keluar.")

while True:
    success, frame = video.read()
    if not success:
        print("Gagal membaca frame dari kamera.")
        break

    frame = cv2.flip(frame, 1)

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

    total_fingers = 0

    if results.multi_hand_landmarks:
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            hand_label = results.multi_handedness[hand_idx].classification[0].label
            fingers_up_for_hand = []
            if hand_label == "Right":
                if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
                    fingers_up_for_hand.append(1)
                else:
                    fingers_up_for_hand.append(0)
            else: 
                if hand_landmarks.landmark[tip_ids[0]].x > hand_landmarks.landmark[tip_ids[0] - 1].x:
                    fingers_up_for_hand.append(1)
                else:
                    fingers_up_for_hand.append(0)
            for i in range(1, 5):
                if hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[tip_ids[i] - 2].y:
                    fingers_up_for_hand.append(1)
                else:
                    fingers_up_for_hand.append(0)
            
            total_fingers += fingers_up_for_hand.count(1)
    cv2.putText(frame, str(total_fingers), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 5)

    cv2.imshow("Deteksi Tangan dan Jari (Tekan 'q' untuk keluar)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print("Membersihkan resource...")
video.release()
cv2.destroyAllWindows()
