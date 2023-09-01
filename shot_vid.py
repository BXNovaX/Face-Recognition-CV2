import cv2

face_classifier = cv2.CascadeClassifier("./models/haarcascade_frontalface_default.xml")
eye_classifier = cv2.CascadeClassifier("./models/haarcascade_eye.xml")
smile_classifier = cv2.CascadeClassifier("./models/haarcascade_smile.xml")

video_capture = cv2.VideoCapture(0)


def detect_faces(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for x, y, w, h in faces:
        cv2.circle(vid, (x + w // 2, y + h // 2), w // 2, (255, 0, 0), 4)
    return faces


def detect_eyes(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    eyes = eye_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for x, y, w, h in eyes:
        cv2.circle(vid, (x + w // 2, y + h // 2), w // 2, (0, 255, 0), 4)
    return eyes


def detect_smiles(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    smiles = smile_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    for x, y, w, h in smiles:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 0, 255), 4)
    return smiles


def main():
    while True:
        result, video_frame = video_capture.read()
        if result is False:
            break

        detect_faces(video_frame)
        detect_eyes(video_frame)
        detect_smiles(video_frame)

        cv2.imshow("My Face Detection Project", video_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
