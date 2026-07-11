import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Your tuned values
lower_yellow = np.array([16, 85, 97])
upper_yellow = np.array([35, 255, 255])

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # --- Processing Pipeline ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Noise reduction
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill small holes
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw center line for reference
    height, width = frame.shape[:2]
    cv2.line(frame, (width//2, 0), (width//2, height), (0, 255, 0), 2)  # Green center line
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:  # Filter small blobs (adjust as needed)
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Calculate center
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Draw center point and text
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"({center_x}, {center_y}) Area:{int(area)}", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Decision example for drivetrain
            if center_x < width // 3:
                direction = "TURN LEFT"
            elif center_x > 2 * width // 3:
                direction = "TURN RIGHT"
            else:
                direction = "CENTER"
            
            cv2.putText(frame, direction, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    
    # Show windows
    cv2.imshow('Original + Detection', frame)
    cv2.imshow('Mask', mask)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()