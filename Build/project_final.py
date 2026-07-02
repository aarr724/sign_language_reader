import os
os.environ['TF_CPP_MIN_LOG_LEVEL']       = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS']      = '0'
os.environ['GLOG_minloglevel']           = '3'
os.environ['MEDIAPIPE_DISABLE_GPU']      = '1'
os.environ['GRPC_VERBOSITY']             = 'ERROR'
os.environ['GRPC_TRACE']                 = ''

import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import math
import cv2
import sys
import traceback
import pyttsx3
import onnxruntime as ort
from cvzone.HandTrackingModule import HandDetector
from string import ascii_uppercase
import enchant
import tkinter as tk
from PIL import Image, ImageTk


ddd = enchant.Dict("en-US")
hd  = HandDetector(maxHands=1)
hd2 = HandDetector(maxHands=1)

offset = 29


class Application:

    def __init__(self):
        self.vs = cv2.VideoCapture(0)
        self.current_image = None


        self.model      = ort.InferenceSession('cnn8grps_rad1_model.onnx')
        self.input_name = self.model.get_inputs()[0].name

        # ── Text-to-speech
        self.speak_engine = pyttsx3.init()
        self.speak_engine.setProperty("rate", 100)
        voices = self.speak_engine.getProperty("voices")
        self.speak_engine.setProperty("voice", voices[0].id)


        self.ct             = {'blank': 0}
        self.blank_flag     = 0
        self.space_flag     = False
        self.next_flag      = True
        self.prev_char      = ""
        self.count          = -1
        self.ten_prev_char  = [" "] * 10
        self.pts            = []

        for ch in ascii_uppercase:
            self.ct[ch] = 0

        print("Loaded model from disk")

        # ── GUI
        self.root = tk.Tk()
        self.root.title(" Sign Language Reader")
        self.root.geometry("1250x720")
        self.root.configure(bg="#0f1117")
        self.root.resizable(False, False)

        self.root.protocol('WM_DELETE_WINDOW', self.destructor)


        self.panel = tk.Label(self.root, bg="#1a1d27", bd=0)
        self.panel.place(x=20, y=20, width=520, height=390)


        self.panel2 = tk.Label(self.root, bg="#1a1d27", bd=0)
        self.panel2.place(x=570, y=20, width=350, height=350)


        tk.Label(
            self.root,
            text="Sign Language Reader",
            bg="#0f1117",
            fg="#00e5ff",
            font=("Courier", 24, "bold")
        ).place(x=20, y=430)


        tk.Label(
            self.root,
            text="Current Character",
            bg="#0f1117",
            fg="#666677",
            font=("Courier", 11, "bold")
        ).place(x=20, y=480)

        self.panel3 = tk.Label(
            self.root,
            text="",
            bg="#0f1117",
            fg="#00ff88",
            font=("Courier", 30, "bold")
        )
        self.panel3.place(x=20, y=500)


        tk.Label(
            self.root,
            text="Sentence",
            bg="#0f1117",
            fg="#666677",
            font=("Courier", 11, "bold")
        ).place(x=250, y=480)

        self.panel5 = tk.Label(
            self.root,
            text="",
            bg="#1a1d27",
            fg="white",
            font=("Courier", 16),
            wraplength=700,
            justify="left",
            anchor="nw",
            padx=15,
            pady=10
        )

        self.panel5.place(x=250, y=510, width=700, height=90)


        tk.Label(
            self.root,
            text="Suggestions",
            bg="#0f1117",
            fg="#ff5577",
            font=("Courier", 12, "bold")
        ).place(x=20, y=620)


        btn_style = {
            "font": ("Courier", 12),
            "bg": "#1a1d27",
            "fg": "#00e5ff",
            "activebackground": "#222233",
            "activeforeground": "white",
            "relief": "flat",
            "bd": 0
        }

        self.b1 = tk.Button(self.root, **btn_style)
        self.b2 = tk.Button(self.root, **btn_style)
        self.b3 = tk.Button(self.root, **btn_style)
        self.b4 = tk.Button(self.root, **btn_style)

        self.b1.place(x=20, y=650, width=200, height=40)
        self.b2.place(x=240, y=650, width=200, height=40)
        self.b3.place(x=460, y=650, width=200, height=40)
        self.b4.place(x=680, y=650, width=200, height=40)


        tk.Button(
            self.root,
            text=" Speak",
            command=self.speak_fun,
            bg="#00e5ff",
            fg="#0f1117",
            activebackground="#00bcd4",
            relief="flat",
            font=("Courier", 13, "bold")
        ).place(x=970, y=620, width=120, height=40)


        tk.Button(
            self.root,
            text="⌫ Clear",
            command=self.clear_fun,
            bg="#ff4466",
            fg="white",
            activebackground="#dd3355",
            relief="flat",
            font=("Courier", 13, "bold")
        ).place(x=1110, y=620, width=120, height=40)


        tk.Label(
            self.root,
            text="Select Gesture",
            bg="#0f1117",
            fg="#666677",
            font=("Courier", 11, "bold")
        ).place(x=970, y=20)

        self.image_panel1 = tk.Label(
            self.root,
            bg="#1a1d27",
            bd=0
        )
        self.image_panel1.place(x=970, y=50, width=240, height=240)

        img1 = Image.open("select.png").resize((240, 240), Image.LANCZOS)
        self.photo1 = ImageTk.PhotoImage(img1)
        self.image_panel1.configure(image=self.photo1)
        self.image_panel1.image = self.photo1


        tk.Label(
            self.root,
            text="Backspace Gesture",
            bg="#0f1117",
            fg="#666677",
            font=("Courier", 11, "bold")
        ).place(x=970, y=320)

        self.image_panel2 = tk.Label(
            self.root,
            bg="#1a1d27",
            bd=0
        )
        self.image_panel2.place(x=970, y=350, width=240, height=240)

        img2 = Image.open("thumbs up.png").resize((240, 240), Image.LANCZOS)
        self.photo2 = ImageTk.PhotoImage(img2)
        self.image_panel2.configure(image=self.photo2)
        self.image_panel2.image = self.photo2


        self.new_window_btn = tk.Button(self.root, text="Sign Language Hint",
                                        font=("Courier", 14),bg="#ff4466",
                                        fg="white",
                                        activebackground="#dd3355",
                                        command=self.open_new_window)
        self.new_window_btn.place(x=570, y=430, width=250, height=35)


        self.status = tk.Label(
            self.root,
            text="ONNX Model Loaded",
            bg="#0a0c12",
            fg="#445566",
            anchor="w",
            font=("Courier", 9)
        )

        self.status.place(x=0, y=700, width=1250, height=20)

        self.str    = " "
        self.ccc    = 0
        self.word   = " "
        self.current_symbol = "C"
        self.word1 = self.word2 = self.word3 = self.word4 = " "

        self.video_loop()

    # ── Main video loop
    def video_loop(self):
        try:
            ok, frame = self.vs.read()
            if not ok or frame is None:
                self.root.after(10, self.video_loop)
                return

            cv2image = cv2.flip(frame, 1)

            hands, cv2image = hd.findHands(cv2image, draw=False, flipType=True)
            cv2image_copy   = np.array(cv2image)
            cv2image_rgb    = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGB)

            self.current_image = Image.fromarray(cv2image_rgb)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            if hands:
                hand        = hands[0]
                x, y, w, h  = hand['bbox']

                y1 = max(0, y - offset)
                y2 = y + h + offset
                x1 = max(0, x - offset)
                x2 = x + w + offset
                image = cv2image_copy[y1:y2, x1:x2]

                if image.size == 0:
                    self.root.after(1, self.video_loop)
                    return

                white = cv2.imread("white.jpg")
                if white is None:
                    white = np.ones((400, 400, 3), dtype=np.uint8) * 255

                hands2, _ = hd2.findHands(image, draw=False, flipType=True)
                self.ccc += 1

                if hands2:
                    hand2    = hands2[0]
                    self.pts = hand2['lmList']

                    os_x = ((400 - w) // 2) - 15
                    os_y = ((400 - h) // 2) - 15

                    self._draw_skeleton(white, os_x, os_y)

                    res = white
                    self.predict(res)

                    self.current_image2 = Image.fromarray(res)
                    imgtk2 = ImageTk.PhotoImage(image=self.current_image2)
                    self.panel2.imgtk = imgtk2
                    self.panel2.config(image=imgtk2)

                    self.panel3.config(text=self.current_symbol)
                    self._update_suggestion_buttons()

            self.panel5.config(text=self.str)

        except Exception:
            print("== video_loop error:", traceback.format_exc())

        finally:
            self.root.after(1, self.video_loop)


    def _draw_skeleton(self, canvas, os_x, os_y):
        p = self.pts

        def pt(i):
            return (p[i][0] + os_x, p[i][1] + os_y)

        connections = [
            list(range(0, 4)),
            list(range(5, 8)),
            list(range(9, 12)),
            list(range(13, 16)),
            list(range(17, 20)),
        ]
        palm_links = [(5, 9), (9, 13), (13, 17), (0, 5), (0, 17)]

        for seg in connections:
            for t in range(len(seg) - 1):
                cv2.line(canvas, pt(seg[t]), pt(seg[t + 1]), (0, 255, 0), 3)
        for a, b in palm_links:
            cv2.line(canvas, pt(a), pt(b), (0, 255, 0), 3)
        for i in range(21):
            cv2.circle(canvas, pt(i), 2, (0, 0, 255), 1)


    def _update_suggestion_buttons(self):
        pairs = [
            (self.b1, self.word1, self.action1),
            (self.b2, self.word2, self.action2),
            (self.b3, self.word3, self.action3),
            (self.b4, self.word4, self.action4),
        ]
        for btn, word, cmd in pairs:
            btn.config(text=word, command=cmd)


    def distance(self, x, y):
        return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)

    def _replace_last_word(self, replacement):
        idx = self.str.rfind(" ")
        self.str = self.str[:idx + 1] + replacement.upper()

    def action1(self): self._replace_last_word(self.word1)
    def action2(self): self._replace_last_word(self.word2)
    def action3(self): self._replace_last_word(self.word3)
    def action4(self): self._replace_last_word(self.word4)

    def speak_fun(self):
        self.speak_engine.say(self.str)
        self.speak_engine.runAndWait()

    def clear_fun(self):
        self.str = " "
        self.word1 = self.word2 = self.word3 = self.word4 = " "

    # ── Prediction (ONNX)
    def predict(self, test_image):
        # prepare input — float32, shape (1, 400, 400, 3)
        inp  = test_image.reshape(1, 400, 400, 3).astype(np.float32)
        prob = np.array(
            self.model.run(None, {self.input_name: inp})[0][0],
            dtype='float32'
        )

        ch1 = int(np.argmax(prob)); prob[ch1] = 0
        ch2 = int(np.argmax(prob)); prob[ch2] = 0
        ch3 = int(np.argmax(prob))

        pl = [ch1, ch2]
        p  = self.pts


        l = [[5,2],[5,3],[3,5],[3,6],[3,0],[3,2],[6,4],[6,1],[6,2],[6,6],[6,7],
             [6,0],[6,5],[4,1],[1,0],[1,1],[6,3],[1,6],[5,6],[5,1],[4,5],[1,4],
             [1,5],[2,0],[2,6],[4,6],[1,0],[5,7],[1,6],[6,1],[7,6],[2,5],[7,1],
             [5,4],[7,0],[7,5],[7,2]]
        if pl in l:
            if p[6][1]<p[8][1] and p[10][1]<p[12][1] and p[14][1]<p[16][1] and p[18][1]<p[20][1]:
                ch1 = 0

        l = [[2,2],[2,1]]
        if pl in l and p[5][0] < p[4][0]:
            ch1 = 0

        pl = [ch1, ch2]
        l = [[0,0],[0,6],[0,2],[0,5],[0,1],[0,7],[5,2],[7,6],[7,1]]
        if pl in l:
            if p[0][0]>p[8][0] and p[0][0]>p[4][0] and p[0][0]>p[12][0] and p[0][0]>p[16][0] and p[0][0]>p[20][0] and p[5][0]>p[4][0]:
                ch1 = 2

        pl = [ch1, ch2]
        if pl in [[6,0],[6,6],[6,2]] and self.distance(p[8], p[16]) < 52:
            ch1 = 2

        pl = [ch1, ch2]
        if pl in [[1,4],[1,5],[1,6],[1,3],[1,0]]:
            if p[6][1]>p[8][1] and p[14][1]<p[16][1] and p[18][1]<p[20][1] and p[0][0]<p[8][0] and p[0][0]<p[12][0] and p[0][0]<p[16][0] and p[0][0]<p[20][0]:
                ch1 = 3

        pl = [ch1, ch2]
        if pl in [[4,6],[4,1],[4,5],[4,3],[4,7]] and p[4][0] > p[0][0]:
            ch1 = 3

        pl = [ch1, ch2]
        if pl in [[5,3],[5,0],[5,7],[5,4],[5,2],[5,1],[5,5]] and p[2][1]+15 < p[16][1]:
            ch1 = 3

        pl = [ch1, ch2]
        if pl in [[6,4],[6,1],[6,2]] and self.distance(p[4], p[11]) > 55:
            ch1 = 4

        pl = [ch1, ch2]
        if pl in [[1,4],[1,6],[1,1]]:
            if self.distance(p[4], p[11]) > 50 and p[6][1]>p[8][1] and p[10][1]<p[12][1] and p[14][1]<p[16][1] and p[18][1]<p[20][1]:
                ch1 = 4

        pl = [ch1, ch2]
        if pl in [[3,6],[3,4]] and p[4][0] < p[0][0]:
            ch1 = 4

        pl = [ch1, ch2]
        if pl in [[2,2],[2,5],[2,4]] and p[1][0] < p[12][0]:
            ch1 = 4

        pl = [ch1, ch2]
        if pl in [[3,6],[3,5],[3,4]]:
            if p[6][1]>p[8][1] and p[10][1]<p[12][1] and p[14][1]<p[16][1] and p[18][1]<p[20][1] and p[4][1]>p[10][1]:
                ch1 = 5

        pl = [ch1, ch2]
        if pl in [[3,2],[3,1],[3,6]]:
            if p[4][1]+17>p[8][1] and p[4][1]+17>p[12][1] and p[4][1]+17>p[16][1] and p[4][1]+17>p[20][1]:
                ch1 = 5

        pl = [ch1, ch2]
        if pl in [[4,4],[4,5],[4,2],[7,5],[7,6],[7,0]] and p[4][0] > p[0][0]:
            ch1 = 5

        pl = [ch1, ch2]
        if pl in [[0,2],[0,6],[0,1],[0,5],[0,0],[0,7],[0,4],[0,3],[2,7]]:
            if p[0][0]<p[8][0] and p[0][0]<p[12][0] and p[0][0]<p[16][0] and p[0][0]<p[20][0]:
                ch1 = 5

        pl = [ch1, ch2]
        if pl in [[5,7],[5,2],[5,6]] and p[3][0] < p[0][0]:
            ch1 = 7

        pl = [ch1, ch2]
        if pl in [[4,6],[4,2],[4,4],[4,1],[4,5],[4,7]] and p[6][1] < p[8][1]:
            ch1 = 7

        pl = [ch1, ch2]
        if pl in [[6,7],[0,7],[0,1],[0,0],[6,4],[6,6],[6,5],[6,1]] and p[18][1] > p[20][1]:
            ch1 = 7

        pl = [ch1, ch2]
        if pl in [[0,4],[0,2],[0,3],[0,1],[0,6]] and p[5][0] > p[16][0]:
            ch1 = 6

        pl = [ch1, ch2]
        if pl == [[7,2]] and p[18][1]<p[20][1] and p[8][1]<p[10][1]:
            ch1 = 6

        pl = [ch1, ch2]
        if pl in [[2,1],[2,2],[2,6],[2,7],[2,0]] and self.distance(p[8], p[16]) > 50:
            ch1 = 6

        pl = [ch1, ch2]
        if pl in [[4,6],[4,2],[4,1],[4,4]] and self.distance(p[4], p[11]) < 60:
            ch1 = 6

        pl = [ch1, ch2]
        if pl in [[1,4],[1,6],[1,0],[1,2]] and p[5][0]-p[4][0]-15 > 0:
            ch1 = 6

        pl = [ch1, ch2]
        if pl in [[5,0],[5,1],[5,4],[5,5],[5,6],[6,1],[7,6],[0,2],[7,1],[7,4],[6,6],[7,2],[5,0],[6,3],[6,4],[7,5],[7,2]]:
            if p[6][1]>p[8][1] and p[10][1]>p[12][1] and p[14][1]>p[16][1] and p[18][1]>p[20][1]:
                ch1 = 1

        pl = [ch1, ch2]
        if pl in [[6,1],[6,0],[0,3],[6,4],[2,2],[0,6],[6,2],[7,6],[4,6],[4,1],[4,2],[0,2],[7,1],[7,4],[6,6],[7,2],[7,5],[7,2]]:
            if p[6][1]<p[8][1] and p[10][1]>p[12][1] and p[14][1]>p[16][1] and p[18][1]>p[20][1]:
                ch1 = 1

        pl = [ch1, ch2]
        if pl in [[6,1],[6,0],[4,2],[4,1],[4,6],[4,4]]:
            if p[10][1]>p[12][1] and p[14][1]>p[16][1] and p[18][1]>p[20][1]:
                ch1 = 1

        pl = [ch1, ch2]
        if pl in [[5,0],[3,4],[3,0],[3,1],[3,5],[5,5],[5,4],[5,1],[7,6]]:
            if p[6][1]>p[8][1] and p[10][1]<p[12][1] and p[14][1]<p[16][1] and p[18][1]<p[20][1] and p[2][0]<p[0][0] and p[4][1]>p[14][1]:
                ch1 = 1

        pl = [ch1, ch2]
        if pl in [[4,1],[4,2],[4,4]]:
            if self.distance(p[4], p[11]) < 50 and p[6][1]>p[8][1] and p[10][1]<p[12][1] and p[14][1]<p[16][1] and p[18][1]<p[20][1]:
                ch1 = 1

        pl = [ch1, ch2]
        if pl in [[3,4],[3,0],[3,1],[3,5],[3,6]]:
            if p[6][1]>p[8][1] and p[10][1]<p[12][1] and p[14][1]<p[16][1] and p[18][1]<p[20][1] and p[2][0]<p[0][0] and p[14][1]<p[4][1]:
                ch1 = 1

        pl = [ch1, ch2]
        if pl in [[6,6],[6,4],[6,1],[6,2]] and p[5][0]-p[4][0]-15 < 0:
            ch1 = 1

        pl = [ch1, ch2]
        if pl in [[5,4],[5,5],[5,1],[0,3],[0,7],[5,0],[0,2],[6,2],[7,5],[7,1],[7,6],[7,7]]:
            if p[6][1]<p[8][1] and p[10][1]<p[12][1] and p[14][1]<p[16][1] and p[18][1]>p[20][1]:
                ch1 = 1

        pl = [ch1, ch2]
        if pl in [[1,5],[1,7],[1,1],[1,6],[1,3],[1,0]]:
            if p[4][0]<p[5][0]+15 and p[6][1]<p[8][1] and p[10][1]<p[12][1] and p[14][1]<p[16][1] and p[18][1]>p[20][1]:
                ch1 = 7

        pl = [ch1, ch2]
        if pl in [[5,5],[5,0],[5,4],[5,1],[4,6],[4,1],[7,6],[3,0],[3,5]]:
            if p[6][1]>p[8][1] and p[10][1]>p[12][1] and p[14][1]<p[16][1] and p[18][1]<p[20][1] and p[4][1]>p[14][1]:
                ch1 = 1

        pl = [ch1, ch2]
        fg = 13
        if pl in [[3,5],[3,0],[3,6],[5,1],[4,1],[2,0],[5,0],[5,5]]:
            if (not (p[0][0]+fg<p[8][0] and p[0][0]+fg<p[12][0] and p[0][0]+fg<p[16][0] and p[0][0]+fg<p[20][0]) and
                not (p[0][0]>p[8][0] and p[0][0]>p[12][0] and p[0][0]>p[16][0] and p[0][0]>p[20][0]) and
                    self.distance(p[4], p[11]) < 50):
                ch1 = 1

        pl = [ch1, ch2]
        if pl in [[5,0],[5,5],[0,1]]:
            if p[6][1]>p[8][1] and p[10][1]>p[12][1] and p[14][1]>p[16][1]:
                ch1 = 1


        if ch1 == 0:
            ch1 = 'S'
            if p[4][0]<p[6][0] and p[4][0]<p[10][0] and p[4][0]<p[14][0] and p[4][0]<p[18][0]:
                ch1 = 'A'
            if p[4][0]>p[6][0] and p[4][0]<p[10][0] and p[4][0]<p[14][0] and p[4][0]<p[18][0] and p[4][1]<p[14][1] and p[4][1]<p[18][1]:
                ch1 = 'T'
            if p[4][1]>p[8][1] and p[4][1]>p[12][1] and p[4][1]>p[16][1] and p[4][1]>p[20][1]:
                ch1 = 'E'
            if p[4][0]>p[6][0] and p[4][0]>p[10][0] and p[4][0]>p[14][0] and p[4][1]<p[18][1]:
                ch1 = 'M'
            if p[4][0]>p[6][0] and p[4][0]>p[10][0] and p[4][1]<p[18][1] and p[4][1]<p[14][1]:
                ch1 = 'N'

        elif ch1 == 2:
            ch1 = 'C' if self.distance(p[12], p[4]) > 42 else 'O'

        elif ch1 == 3:
            ch1 = 'G' if self.distance(p[8], p[12]) > 72 else 'H'

        elif ch1 == 7:
            ch1 = 'Y' if self.distance(p[8], p[4]) > 42 else 'J'

        elif ch1 == 4:
            ch1 = 'L'

        elif ch1 == 6:
            ch1 = 'X'

        elif ch1 == 5:
            if p[4][0]>p[12][0] and p[4][0]>p[16][0] and p[4][0]>p[20][0]:
                ch1 = 'Z' if p[8][1] < p[5][1] else 'Q'
            else:
                ch1 = 'P'

        elif ch1 == 1:
            idx_up = p[6][1]  > p[8][1]
            mid_up = p[10][1] > p[12][1]
            rng_up = p[14][1] > p[16][1]
            pnk_up = p[18][1] > p[20][1]

            if   idx_up and mid_up and rng_up and pnk_up:                    ch1 = 'B'
            elif idx_up and not mid_up and not rng_up and not pnk_up:        ch1 = 'D'
            elif not idx_up and mid_up and rng_up and pnk_up:                ch1 = 'F'
            elif not idx_up and not mid_up and not rng_up and pnk_up:        ch1 = 'I'
            elif idx_up and mid_up and rng_up and not pnk_up:                ch1 = 'W'
            elif idx_up and mid_up and not rng_up and not pnk_up and p[4][1]<p[9][1]:
                ch1 = 'K'
            elif idx_up and mid_up and not rng_up and not pnk_up and (self.distance(p[8],p[12])-self.distance(p[6],p[10])) < 8:
                ch1 = 'U'
            elif idx_up and mid_up and not rng_up and not pnk_up and (self.distance(p[8],p[12])-self.distance(p[6],p[10])) >= 8 and p[4][1]>p[9][1]:
                ch1 = 'V'
            elif idx_up and mid_up and not rng_up and not pnk_up and p[8][0]>p[12][0]:
                ch1 = 'R'

        # ── special gestures ──────────────────────────────────────────────────
        if ch1 in (1, 'E', 'S', 'X', 'Y', 'B'):
            if p[6][1]>p[8][1] and p[10][1]<p[12][1] and p[14][1]<p[16][1] and p[18][1]>p[20][1]:
                ch1 = " "

        if ch1 in ('E', 'Y', 'B'):
            if p[4][0]<p[5][0] and p[6][1]>p[8][1] and p[10][1]>p[12][1] and p[14][1]>p[16][1] and p[18][1]>p[20][1]:
                ch1 = "next"

        if ch1 in {'Next', 'B', 'C', 'H', 'F', 'X'}:
            if (p[0][0]>p[8][0] and p[0][0]>p[12][0] and p[0][0]>p[16][0] and p[0][0]>p[20][0] and
                    p[4][1]<p[8][1] and p[4][1]<p[12][1] and p[4][1]<p[16][1] and p[4][1]<p[20][1] and
                    p[4][1]<p[6][1] and p[4][1]<p[10][1] and p[4][1]<p[14][1] and p[4][1]<p[18][1]):
                ch1 = 'Backspace'


        if ch1 == "next" and self.prev_char != "next":
            prev2 = self.ten_prev_char[(self.count - 2) % 10]
            prev0 = self.ten_prev_char[(self.count - 0) % 10]
            if prev2 != "next":
                if prev2 == "Backspace":
                    self.str = self.str[:-1]
                else:
                    self.str += prev2
            else:
                if prev0 != "Backspace":
                    self.str += prev0

        if ch1 == "  " and self.prev_char != "  ":
            self.str += "  "

        self.prev_char = ch1
        self.current_symbol = ch1
        self.count += 1
        self.ten_prev_char[self.count % 10] = ch1


        if self.str.strip():
            st   = self.str.rfind(" ")
            word = self.str[st + 1:]
            self.word = word
            if word.strip():
                suggestions = ddd.suggest(word)
                self.word1 = suggestions[0] if len(suggestions) >= 1 else " "
                self.word2 = suggestions[1] if len(suggestions) >= 2 else " "
                self.word3 = suggestions[2] if len(suggestions) >= 3 else " "
                self.word4 = suggestions[3] if len(suggestions) >= 4 else " "
            else:
                self.word1 = self.word2 = self.word3 = self.word4 = " "

    # ── Cleanup
    def destructor(self):
        print(self.ten_prev_char)
        self.root.destroy()
        self.vs.release()
        cv2.destroyAllWindows()

    def open_new_window(self):
        new_win = tk.Toplevel(self.root)
        new_win.title("Sign Language")
        new_win.geometry("1000x800")
        img = Image.open("sign.jpg").resize((800, 600), Image.LANCZOS)
        self._new_win_photo = ImageTk.PhotoImage(img)

        img_label = tk.Label(new_win, image=self._new_win_photo)
        img_label.image = self._new_win_photo
        img_label.place(x=10, y=10, width=800, height=600)



print("Starting Application...")
Application().root.mainloop()