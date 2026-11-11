import cv2
import numpy as np
import time

LOWER_FIRE = np.array([0, 80, 180])
UPPER_FIRE = np.array([35, 250, 255])

MIN_FIRE_AREA = 1500          
FLICKER_THRESHOLD = 4000      
CONFIDENCE_THRESH = 0.8        

def calculate_confidence(fire_pixels, flicker_pixels):
  
    conf_color = min(fire_pixels / (MIN_FIRE_AREA * 2), 1.0)
    conf_flicker = min(flicker_pixels / (FLICKER_THRESHOLD * 2), 1.0)
    return (conf_color * 0.7 + conf_flicker * 0.3)

cap = cv2.VideoCapture(0)
previous_mask = None

fps_counter, start_time = 0, time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, LOWER_FIRE, UPPER_FIRE)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fire_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > fire_area:
            fire_area = area
            largest_cnt = cnt

    fire_pixels = cv2.countNonZero(mask)
    fire_detected = False
    confidence = 0
    flicker_pixels = 0

    if previous_mask is not None:
        diff = cv2.absdiff(mask, previous_mask)
        flicker_pixels = cv2.countNonZero(diff)
    previous_mask = mask.copy()


    condition_color = fire_pixels > MIN_FIRE_AREA

    condition_flicker = flicker_pixels > FLICKER_THRESHOLD

    confidence = calculate_confidence(fire_pixels, flicker_pixels)
    if condition_color and condition_flicker and confidence > CONFIDENCE_THRESH:
        fire_detected = True


    if fire_detected and fire_area > MIN_FIRE_AREA:
        x, y, w, h = cv2.boundingRect(largest_cnt)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, f"FIRE ({confidence*100:.1f}%)", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    else:
        cv2.putText(frame, f"No Fire ({confidence*100:.1f}%)", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)


    fps_counter += 1
    if time.time() - start_time > 1.0:
        fps = fps_counter / (time.time() - start_time)
        fps_counter, start_time = 0, time.time()
        cv2.putText(frame, f"FPS: {fps:.2f}", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Fire Mask", mask)
    cv2.imshow("Fire Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()