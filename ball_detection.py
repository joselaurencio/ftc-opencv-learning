import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Tuned HSV
lower_yellow = np.array([16, 85, 97])
upper_yellow = np.array([35, 255, 255])

# Smoothing variables
center_x_history = []
SMOOTHING_WINDOW = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    original = frame.copy()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Enhanced noise reduction
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    height, width = frame.shape[:2]
    cv2.line(frame, (width//3, 0), (width//3, height), (0, 165, 255), 2)      # Left zone
    cv2.line(frame, (2*width//3, 0), (2*width//3, height), (0, 165, 255), 2) # Right zone
    
    best_center_x = None
    best_center_y = None
    biggest_area = 0
    best_box = None
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # stricter filter
            x, y, w, h = cv2.boundingRect(contour)
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Draw
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(frame, (center_x, center_y), 6, (0, 0, 255), -1)
            
            if area > biggest_area:
                biggest_area = area
                best_center_x = center_x
                best_center_y = center_y
                best_box = (x, y, w, h)
    
    # Smoothing
    if best_center_x is not None:
        center_x_history.append(best_center_x)
        if len(center_x_history) > SMOOTHING_WINDOW:
            center_x_history.pop(0)
        
        smoothed_x = int(sum(center_x_history) / len(center_x_history))
        
        # Draw smoothed position
        cv2.circle(frame, (smoothed_x, best_center_y), 8, (255, 0, 0), -1)
        cv2.putText(frame, f"Smoothed X: {smoothed_x}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Decision zones (more forgiving)
        if smoothed_x < width // 3:
            decision = "TURN LEFT"
            color = (0, 0, 255)
        elif smoothed_x > 2 * width // 3:
            decision = "TURN RIGHT"
            color = (0, 0, 255)
        else:
            decision = "CENTER - FORWARD"
            color = (0, 255, 0)
        
        cv2.putText(frame, decision, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
        cv2.putText(frame, f"Area: {int(biggest_area)}", (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        cv2.putText(frame, "NO TARGET", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    
    cv2.imshow('Improved Yellow Ball Detection', frame)
    cv2.imshow('Mask', mask)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):  # Press 's' to save current frame + mask
        cv2.imwrite('debug_frame.png', original)
        cv2.imwrite('debug_mask.png', mask)
        print("Saved debug images")

cap.release()
cv2.destroyAllWindows()