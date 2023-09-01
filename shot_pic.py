from cv2 import VideoCapture, imshow, imwrite, waitKey, destroyWindow

cam_port = 0
cam = VideoCapture(cam_port)

result, image = cam.read()

if result:
    imshow("Captured Image", image)
    waitKey(0)
    destroyWindow("GeeksForGeeks")

else:
    print("No image detected. Please! try again")
