import cv2
import numpy as np

cap = cv2.VideoCapture(0)
lower_yellow = np.array([16, 85, 97])
upper_yellow = np.array([35, 255, 255])

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    original = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Stronger noise reduction
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Optional: Blur the mask a bit
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    height, width = frame.shape[:2]
    cv2.line(frame, (width//2, 0), (width//2, height), (0, 255, 0), 2)   # Center line
    
    best_center_x = None
    biggest_area = 0
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 800:                     # Increased threshold
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Draw on frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
            cv2.circle(frame, (center_x, center_y), 8, (0, 0, 255), -1)
            cv2.putText(frame, f"Area: {int(area)}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Track the largest ball (most relevant)
            if area > biggest_area:
                biggest_area = area
                best_center_x = center_x
    
    # Decision based on largest ball
    if best_center_x is not None:
        if best_center_x < width // 3:
            decision = "TURN LEFT"
            color = (0, 0, 255)
        elif best_center_x > 2 * width // 3:
            decision = "TURN RIGHT"
            color = (0, 0, 255)
        else:
            decision = "CENTER / GO FORWARD"
            color = (0, 255, 0)
        
        cv2.putText(frame, decision, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame, f"Largest ball at x={best_center_x}", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.imshow('Detection', frame)
    cv2.imshow('Mask', mask)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()