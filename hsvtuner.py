import cv2
import numpy as np

def nothing(x):
    pass

# Load your webcam (or change to cv2.imread('your_image.jpg') for static testing)
cap = cv2.VideoCapture(0)

cv2.namedWindow('HSV Tuner')
cv2.createTrackbar('H Min', 'HSV Tuner', 20, 179, nothing)
cv2.createTrackbar('H Max', 'HSV Tuner', 35, 179, nothing)
cv2.createTrackbar('S Min', 'HSV Tuner', 80, 255, nothing)
cv2.createTrackbar('S Max', 'HSV Tuner', 255, 255, nothing)
cv2.createTrackbar('V Min', 'HSV Tuner', 80, 255, nothing)
cv2.createTrackbar('V Max', 'HSV Tuner', 255, 255, nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Get current trackbar positions
    h_min = cv2.getTrackbarPos('H Min', 'HSV Tuner')
    h_max = cv2.getTrackbarPos('H Max', 'HSV Tuner')
    s_min = cv2.getTrackbarPos('S Min', 'HSV Tuner')
    s_max = cv2.getTrackbarPos('S Max', 'HSV Tuner')
    v_min = cv2.getTrackbarPos('V Min', 'HSV Tuner')
    v_max = cv2.getTrackbarPos('V Max', 'HSV Tuner')
    
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    
    mask = cv2.inRange(hsv, lower, upper)
    result = cv2.bitwise_and(frame, frame, mask=mask)
    
    # Show everything
    cv2.imshow('Original', frame)
    cv2.imshow('Mask', mask)
    cv2.imshow('Result', result)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(f"Final HSV range -> Lower: {lower} | Upper: {upper}")
        break

cap.release()
cv2.destroyAllWindows()