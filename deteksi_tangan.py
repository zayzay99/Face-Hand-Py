import cv2
import mediapipe as mp

# --- Inisialisasi MediaPipe Hands ---
mp_hands = mp.solutions.hands
# Parameter:
# static_image_mode=False -> Deteksi pada video stream
# max_num_hands=1 -> Hanya deteksi satu tangan untuk menyederhanakan
# min_detection_confidence=0.5 -> Ambang batas kepercayaan deteksi
# min_tracking_confidence=0.5 -> Ambang batas kepercayaan pelacakan
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Utilitas untuk menggambar landmark tangan
mp_drawing = mp.solutions.drawing_utils

# --- Inisialisasi Video Capture ---
video = cv2.VideoCapture(0)
if not video.isOpened():
    print("Tidak bisa membuka kamera.")
    exit()

# ID landmark untuk ujung setiap jari
# Lihat gambar landmark di dokumentasi MediaPipe untuk referensi
# https://google.github.io/mediapipe/solutions/hands.html#hand-landmark-model
tip_ids = [4, 8, 12, 16, 20]

print("Memulai deteksi tangan... Tekan 'q' untuk keluar.")

while True:
    # Baca frame dari video
    success, frame = video.read()
    if not success:
        print("Gagal membaca frame dari kamera.")
        break

    # Balik frame secara horizontal (seperti cermin) agar lebih intuitif
    frame = cv2.flip(frame, 1)

    # Konversi warna dari BGR (OpenCV) ke RGB (MediaPipe)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Proses gambar untuk mendeteksi tangan
    results = hands.process(image_rgb)

    # Jika tangan terdeteksi
    if results.multi_hand_landmarks:
        # Ambil landmark dari tangan pertama yang terdeteksi
        hand_landmarks = results.multi_hand_landmarks[0]

        # Gambar landmark dan koneksinya pada frame
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # --- Logika Menghitung Jari ---
        fingers_up = []
        
        # 1. Ibu Jari (Thumb)
        # Logikanya sedikit berbeda: cek posisi horizontal (sumbu x)
        # Jika ujung ibu jari lebih kiri dari pangkalnya (untuk tangan kanan), berarti terangkat
        if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
            fingers_up.append(1)
        else:
            fingers_up.append(0)

        # 2. Empat Jari Lainnya (Telunjuk, Tengah, Manis, Kelingking)
        # Logikanya: cek posisi vertikal (sumbu y)
        # Jika ujung jari lebih atas dari ruas di bawahnya, berarti terangkat
        for i in range(1, 5):
            if hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[tip_ids[i] - 2].y:
                fingers_up.append(1)
            else:
                fingers_up.append(0)
        
        # Hitung total jari yang terangkat
        total_fingers = fingers_up.count(1)

        # Tampilkan jumlah jari di layar
        cv2.putText(frame, str(total_fingers), (50, 100), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 5)

    # Tampilkan frame hasil
    cv2.imshow("Deteksi Tangan dan Jari (Tekan 'q' untuk keluar)", frame)

    # Tunggu tombol 'q' ditekan untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Hentikan video dan tutup semua jendela
print("Membersihkan resource...")
video.release()
cv2.destroyAllWindows()
