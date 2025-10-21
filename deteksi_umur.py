import cv2
import numpy as np
import os
import sys

FACE_PROTO = "model/deploy.prototxt.txt"
FACE_MODEL = "model/res10_300x300_ssd_iter_140000.caffemodel"
AGE_PROTO = "model/age_deploy.prototxt"
AGE_MODEL = "model/age_net.caffemodel"
MODEL_MEAN_VALUES = (78.42633776, 87.76891437, 114.89584775)

AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]

CONFIDENCE_THRESHOLD = 0.5  
PADDING = 20                


def load_models():
    for path in [FACE_PROTO, FACE_MODEL, AGE_PROTO, AGE_MODEL]:
        if not os.path.exists(path):
            print(f"[ERROR] File model tidak ditemukan di: {path}")
            sys.exit("Silakan unduh file model dan letakkan di folder 'model'.")

    print("[INFO] Memuat model deteksi wajah...")
    face_net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
    if face_net.empty():
        print(f"[ERROR] Gagal memuat model deteksi wajah dari {FACE_MODEL} dan {FACE_PROTO}")
        sys.exit("Pastikan file model tidak rusak atau formatnya tidak didukung.")

    print("[INFO] Memuat model estimasi umur...")
    age_net = cv2.dnn.readNet(AGE_MODEL, AGE_PROTO)
    if age_net.empty():
        print(f"[ERROR] Gagal memuat model estimasi umur dari {AGE_MODEL} dan {AGE_PROTO}")
        sys.exit("Pastikan file model tidak rusak atau formatnya tidak didukung.")

    return face_net, age_net


def process_frame(frame, face_net, age_net):
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))

    face_net.setInput(blob)
    detections = face_net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > CONFIDENCE_THRESHOLD:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            face = frame[max(0, startY - PADDING):min(endY + PADDING, h - 1),
                         max(0, startX - PADDING):min(endX + PADDING, w - 1)]

            if face.shape[0] < 20 or face.shape[1] < 20:
                continue
            face_blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            age_net.setInput(face_blob)
            preds = age_net.forward()
            
            # Logika untuk model lama (menggunakan rentang umur)
            age_range = AGE_BUCKETS[preds[0].argmax()]

            text = f"Umur: {age_range}"

            y = startY - 10 if startY - 10 > 10 else startY + 10

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame


def main():
    face_net, age_net = load_models()
    print("[INFO] Memulai stream video dari webcam...")
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        print("[ERROR] Tidak bisa membuka kamera.")
        return
    while True:
        has_frame, frame = video.read()
        if not has_frame:
            print("[INFO] Stream video berakhir. Keluar...")
            break
        processed_frame = process_frame(frame, face_net, age_net)
        cv2.imshow("Deteksi Wajah dan Umur (Tekan 'q' untuk keluar)", processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("[INFO] Membersihkan resource...")
    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
