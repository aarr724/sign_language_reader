import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import os
import traceback

capture = cv2.VideoCapture(0)
hd  = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)


DATASET_DIR = "C:\\Users\\dell\\IdeaProjects\\PythonProject1\\AtoZ_3.1\\"
offset  = 15
step    = 1
flag    = False
suv     = 0
c_dir   = 'A'
count   = len(os.listdir(DATASET_DIR + c_dir + "\\"))

# create white canvas once
white_path = "white.jpg"
white_base = np.ones((400, 400, 3), dtype=np.uint8) * 255
cv2.imwrite(white_path, white_base)

skeleton1 = None

while True:
    try:
        _, frame = capture.read()
        frame = cv2.flip(frame, 1)


        hands, frame = hd.findHands(frame, draw=False, flipType=True)

        white = cv2.imread(white_path)
        if white is None:
            white = np.ones((400, 400, 3), dtype=np.uint8) * 255

        if hands:
            hand       = hands[0]                    
            x, y, w, h = hand['bbox']

            y1 = max(0, y - offset)
            y2 = y + h + offset
            x1 = max(0, x - offset)
            x2 = x + w + offset
            image = np.array(frame[y1:y2, x1:x2])

            if image.size == 0:
                cv2.imshow("frame", frame)
                cv2.waitKey(1)
                continue


            hands2, imz = hd2.findHands(image, draw=True, flipType=True)

            if hands2:
                hand2 = hands2[0]
                pts   = hand2['lmList']

                os_x  = ((400 - w) // 2) - 15
                os_y  = ((400 - h) // 2) - 15

                def pt(i):
                    return (pts[i][0] + os_x, pts[i][1] + os_y)


                for t in range(0, 4): cv2.line(white, pt(t),    pt(t+1),  (0,255,0), 3)
                for t in range(5, 8): cv2.line(white, pt(t),    pt(t+1),  (0,255,0), 3)
                for t in range(9,12): cv2.line(white, pt(t),    pt(t+1),  (0,255,0), 3)
                for t in range(13,16):cv2.line(white, pt(t),    pt(t+1),  (0,255,0), 3)
                for t in range(17,20):cv2.line(white, pt(t),    pt(t+1),  (0,255,0), 3)

                cv2.line(white, pt(5),  pt(9),  (0,255,0), 3)
                cv2.line(white, pt(9),  pt(13), (0,255,0), 3)
                cv2.line(white, pt(13), pt(17), (0,255,0), 3)
                cv2.line(white, pt(0),  pt(5),  (0,255,0), 3)
                cv2.line(white, pt(0),  pt(17), (0,255,0), 3)

                for i in range(21):
                    cv2.circle(white, pt(i), 2, (0,0,255), 1)

                skeleton1 = np.array(white)
                cv2.imshow("Skeleton", skeleton1)


        frame = cv2.putText(frame,
                            f"dir={c_dir}  count={count}  collecting={'ON' if flag else 'OFF'}",
                            (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)
        cv2.imshow("frame", frame)

        interrupt = cv2.waitKey(1)

        # esc=quit and n= next and a= start/stop collection
        if interrupt & 0xFF == 27:
            break


        if interrupt & 0xFF == ord('n'):
            c_dir = chr(ord(c_dir) + 1)
            if ord(c_dir) == ord('Z') + 1:
                c_dir = 'A'
            flag  = False
            count = len(os.listdir(DATASET_DIR + c_dir + "\\"))
            print(f"Switched to: {c_dir}  count={count}")

        if interrupt & 0xFF == ord('a'):
            if flag:
                flag = False
                print("Collection STOPPED")
            else:
                suv  = 0
                flag = True
                print("Collection STARTED")

        if flag:
            if suv >= 180:
                flag = False
                print(f"Done collecting {c_dir} — 180 images saved")
            elif step % 3 == 0 and skeleton1 is not None:
                save_path = DATASET_DIR + c_dir + "\\" + str(count) + ".jpg"
                cv2.imwrite(save_path, skeleton1)
                count += 1
                suv   += 1
                print(f"Saved {save_path}  ({suv}/180)")
            step += 1

    except Exception:
        print("==", traceback.format_exc())

capture.release()
cv2.destroyAllWindows()