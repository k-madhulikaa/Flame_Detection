import cv2
import numpy as np

LOWER_FIRE = np.array([5, 160, 150])
UPPER_FIRE = np.array([30, 255, 255])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, LOWER_FIRE, UPPER_FIRE)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                            np.ones((5,5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                            np.ones((7,7), np.uint8))

    fire_pixels = cv2.countNonZero(mask)

    cv2.putText(frame, f"Flame Area: {fire_pixels}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0,0,255), 2)

    cv2.imshow("Fire Mask", mask)
    cv2.imshow("Flame Detection", frame)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
