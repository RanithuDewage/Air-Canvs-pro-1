import cv2
import mediapipe as mp
import numpy as np
import time
import math
import os
from dataclasses import dataclass
from collections import deque
from pathlib import Path


# =========================
# CONFIG
# =========================
WIDTH, HEIGHT = 1280, 720
TOP_H = 82
WINDOW_NAME = "Air canvas drawing - Developed by RUSH"

PALETTE = [
    ("Red",    (0, 0, 255)),
    ("Green",  (0, 255, 0)),
    ("Blue",   (255, 0, 0)),
    ("Yellow", (0, 255, 255)),
    ("Purple", (255, 0, 255)),
    ("Orange", (0, 140, 255)),
    ("Cyan",   (255, 255, 0)),
    ("White",  (255, 255, 255)),
]

BRUSH_PRESETS = [4, 8, 14, 22, 32]


@dataclass
class Button:
    name: str
    rect: tuple
    kind: str
    value: object


@dataclass
class Action:
    kind: str
    points: list
    color: tuple = (0, 0, 0)
    thickness: int = 1
    shape_type: str = ""


# =========================
# HAND TRACKER
# =========================
class HandTracker:
    def __init__(self, max_hands=2, det_conf=0.78, track_conf=0.78, model_complexity=1):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=model_complexity,
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles
        self.results = None

    def find_hands(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        self.results = self.hands.process(rgb)
        return self.results

    def get_landmarks(self, frame, results=None, hand_index=0, draw=True):
        if results is None:
            results = self.results

        lm_list = []

        if results and results.multi_hand_landmarks:
            if hand_index < len(results.multi_hand_landmarks):
                hand = results.multi_hand_landmarks[hand_index]

                if draw:
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_styles.get_default_hand_landmarks_style(),
                        self.mp_styles.get_default_hand_connections_style(),
                    )

                h, w, _ = frame.shape
                for idx, lm in enumerate(hand.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append((idx, cx, cy))

        return lm_list

    def get_hand_label(self, results=None, hand_index=0):
        if results is None:
            results = self.results

        if results and results.multi_handedness:
            if hand_index < len(results.multi_handedness):
                return results.multi_handedness[hand_index].classification[0].label

        return None

    def fingers_up(self, lm_list, hand_label=None):
        if not lm_list or len(lm_list) < 21:
            return [False, False, False, False, False]

        fingers = [False, False, False, False, False]

        # Thumb
        if hand_label == "Right":
            fingers[0] = lm_list[4][1] < lm_list[3][1]
        elif hand_label == "Left":
            fingers[0] = lm_list[4][1] > lm_list[3][1]
        else:
            fingers[0] = lm_list[4][1] < lm_list[3][1]

        # Other fingers
        fingers[1] = lm_list[8][2] < lm_list[6][2]
        fingers[2] = lm_list[12][2] < lm_list[10][2]
        fingers[3] = lm_list[16][2] < lm_list[14][2]
        fingers[4] = lm_list[20][2] < lm_list[18][2]

        return fingers

    @staticmethod
    def finger_count(fingers):
        return sum(1 for f in fingers if f)


# =========================
# GESTURE ENGINE
# =========================
class GestureEngine:
    @staticmethod
    def distance(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def pinch_info(self, lm_list):
        if not lm_list or len(lm_list) < 10:
            return 9999.0, 9999.0

        thumb = (lm_list[4][1], lm_list[4][2])
        index = (lm_list[8][1], lm_list[8][2])
        wrist = (lm_list[0][1], lm_list[0][2])
        middle_mcp = (lm_list[9][1], lm_list[9][2])

        tip_dist = self.distance(thumb, index)
        palm_scale = max(self.distance(wrist, middle_mcp), 1.0)
        ratio = tip_dist / palm_scale
        return tip_dist, ratio

    def recognize(self, fingers, lm_list):
        count = sum(1 for f in fingers if f)
        tip_dist, pinch_ratio = self.pinch_info(lm_list)

        pinch = pinch_ratio < 0.45 and count <= 2

        if count == 0:
            return "FIST", pinch_ratio

        if count == 5:
            return "Hand", pinch_ratio

        if pinch:
            return "PINCH", pinch_ratio

        if fingers[1] and fingers[2] and fingers[3] and not fingers[0] and not fingers[4]:
            return "THREE", pinch_ratio

        if fingers[1] and fingers[2] and not fingers[0] and not fingers[3] and not fingers[4]:
            return "TWO", pinch_ratio

        if fingers[1] and not fingers[0] and not fingers[2] and not fingers[3] and not fingers[4]:
            return "POINT", pinch_ratio

        return f"CUSTOM_{count}", pinch_ratio


# =========================
# APP
# =========================
class VirtualPainter:
    def __init__(self):
        if os.name == "nt":
            self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            raise SystemExit("Check your camera index or permissions")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.tracker = HandTracker(max_hands=2, det_conf=0.78, track_conf=0.78)
        self.gesture_engine = GestureEngine()

        self.canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        self.buttons = self.build_buttons()

        self.current_color = (255, 0, 255)
        self.current_brush = 8
        self.tool = "pen"
        self.global_mode = "free"
        self.shape_type = "line"

        self.hand_states = {}
        self.actions = []

        self.mode = "IDLE"
        self.message = "Ready"
        self.message_until = 0
        self.select_cooldown_until = 0
        self.save_requested = False
        self.help_visible = False
        self.pinch_msg_until = 0

        self.recording = False
        self.video_writer = None
        self.record_path = None

        self.fps = 0
        self.fps_frame_count = 0
        self.fps_last_time = time.time()

        self.save_dir = Path("captures")
        self.save_dir.mkdir(exist_ok=True)

        self.clear_hold_seconds = 1.2

    # =========================
    # HAND STATE
    # =========================
    def new_hand_state(self):
        return {
            "action": None,
            "sig": None,
            "buffer": deque(maxlen=6),
            "shape_active": False,
            "shape_start": None,
            "shape_end": None,
            "shape_color": (255, 255, 255),
            "shape_thickness": 8,
            "shape_type": "line",
            "fist_start": None,
            "last_seen": 0.0,
        }

    # =========================
    # UI / BUTTONS
    # =========================
    def build_buttons(self):
        buttons = []
        x = 10
        y = 18
        bw = 44
        bh = 44
        gap = 8

        for name, color in PALETTE:
            buttons.append(Button(name=name, rect=(x, y, x + bw, y + bh), kind="color", value=color))
            x += bw + gap

        x += 12

        for size in BRUSH_PRESETS:
            buttons.append(Button(name=str(size), rect=(x, y, x + bw, y + bh), kind="brush", value=size))
            x += bw + gap

        x += 16

        buttons.append(Button(name="Pen", rect=(x, y, x + 78, y + bh), kind="tool", value="pen"))
        x += 86
        buttons.append(Button(name="Eraser", rect=(x, y, x + 92, y + bh), kind="tool", value="eraser"))
        x += 100
        buttons.append(Button(name="Shape", rect=(x, y, x + 86, y + bh), kind="mode", value="shape"))
        x += 94
        buttons.append(Button(name="Undo", rect=(x, y, x + 78, y + bh), kind="action", value="undo"))
        x += 86
        buttons.append(Button(name="Clear", rect=(x, y, x + 82, y + bh), kind="action", value="clear"))

        return buttons

    def inside(self, pt, rect):
        x, y = pt
        x1, y1, x2, y2 = rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def show_message(self, text, seconds=1.5):
        self.message = text
        self.message_until = time.time() + seconds

    def draw_center_text(self, img, text, rect, scale=0.52, color=(255, 255, 255), thickness=1):
        x1, y1, x2, y2 = rect
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        x = x1 + (x2 - x1 - tw) // 2
        y = y1 + (y2 - y1 + th) // 2
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)

    def draw_toolbar(self, frame):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (WIDTH, TOP_H), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

        cv2.putText(frame, "RUSHX AIR CANVAS", (10, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 1, cv2.LINE_AA)

        for btn in self.buttons:
            x1, y1, x2, y2 = btn.rect

            active = False
            fill = (60, 60, 60)
            border = (220, 220, 220)

            if btn.kind == "color":
                fill = btn.value
                active = (self.tool == "pen" and self.current_color == btn.value)
            elif btn.kind == "brush":
                fill = (70, 70, 70)
                active = (self.current_brush == btn.value)
            elif btn.kind == "tool":
                fill = (80, 80, 80) if btn.value == "pen" else (110, 110, 110)
                active = (self.tool == btn.value)
            elif btn.kind == "mode":
                fill = (160, 80, 0)
                active = (self.global_mode == "shape")
            elif btn.kind == "action":
                if btn.value == "undo":
                    fill = (0, 140, 255)
                elif btn.value == "clear":
                    fill = (0, 0, 200)
                elif btn.value == "save":
                    fill = (0, 150, 0)

            if active:
                border = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), fill, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), border, 2)

            if btn.kind in ("brush", "tool", "action", "mode"):
                self.draw_center_text(frame, btn.name, btn.rect, scale=0.50, color=(255, 255, 255), thickness=2)

        cv2.circle(frame, (WIDTH - 34, 41), 16, self.current_color, -1)
        cv2.circle(frame, (WIDTH - 34, 41), 16, (255, 255, 255), 2)

        if self.recording:
            cv2.rectangle(frame, (WIDTH - 120, 10), (WIDTH - 18, 34), (0, 0, 180), -1)
            cv2.putText(frame, "REC", (WIDTH - 93, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.line(frame, (0, TOP_H), (WIDTH, TOP_H), (90, 90, 90), 1)

    def draw_badge(self, frame, x, y, text, color, text_color=(255, 255, 255)):
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        pad_x = 10
        pad_y = 8
        cv2.rectangle(frame, (x, y), (x + tw + pad_x * 2, y + th + pad_y), color, -1)
        cv2.putText(frame, text, (x + pad_x, y + th + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, text_color, 1, cv2.LINE_AA)
        return x + tw + pad_x * 2 + 8

    def draw_status(self, frame, hands_count):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, HEIGHT - 46), (WIDTH, HEIGHT), (18, 18, 18), -1)
        cv2.addWeighted(overlay, 0.50, frame, 0.50, 0, frame)

        x = 12
        y = HEIGHT - 38

        track_state = "ON" if hands_count > 0 else "OFF"
        draw_state = "ON" if self.tool == "pen" else "OFF"
        erase_state = "ON" if self.tool == "eraser" else "OFF"
        shape_state = "ON" if self.global_mode == "shape" else "OFF"
        rec_state = "ON" if self.recording else "OFF"

        x = self.draw_badge(frame, x, y, f"TRACK {track_state}", (0, 140, 0) if track_state == "ON" else (120, 0, 0))
        x = self.draw_badge(frame, x, y, f"DRAW {draw_state}", (0, 140, 0) if draw_state == "ON" else (90, 90, 90))
        x = self.draw_badge(frame, x, y, f"ERASE {erase_state}", (180, 80, 0) if erase_state == "ON" else (90, 90, 90))
        x = self.draw_badge(frame, x, y, f"SHAPE {shape_state}", (160, 80, 0) if shape_state == "ON" else (90, 90, 90))
        x = self.draw_badge(frame, x, y, f"REC {rec_state}", (0, 0, 180) if rec_state == "ON" else (90, 90, 90))
        x = self.draw_badge(frame, x, y, f"HANDS {hands_count}", (60, 60, 60))

        cv2.putText(frame, f"Mode: {self.mode}   Brush: {self.current_brush}   Shape: {self.shape_type}",
                    (12, HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (190, 190, 190), 1, cv2.LINE_AA)

        if time.time() < self.message_until:
            cv2.putText(frame, self.message, (WIDTH // 2 - 150, HEIGHT - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.putText(frame, f"FPS: {self.fps:.1f}", (WIDTH - 125, HEIGHT - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 120), 2, cv2.LINE_AA)

    def draw_help(self, frame):
        if not self.help_visible:
            return

        lines = [
            "1 finger = draw",
            "2 fingers = toolbar select",
            "3 fingers = erase",
            "5 fingers = pause",
            "Pinch = brush size",
            "Fist hold = clear",
            "m = shape mode",
            "l / b / o = line / rect / circle",
            "r = record on/off",
            "s = save snapshot",
            "q = quit",
        ]

        x, y = 930, 95
        w, h = 320, 250
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 80, 80), 2)
        cv2.putText(frame, "Help", (x + 12, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2, cv2.LINE_AA)

        yy = y + 48
        for ln in lines:
            cv2.putText(frame, ln, (x + 14, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
            yy += 20

    # =========================
    # RECORDING / SAVE
    # =========================
    def toggle_recording(self):
        if not self.recording:
            ts = time.strftime("%Y%m%d_%H%M%S")
            self.record_path = self.save_dir / f"record_{ts}.avi"
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            self.video_writer = cv2.VideoWriter(str(self.record_path), fourcc, 20.0, (WIDTH, HEIGHT))

            if not self.video_writer.isOpened():
                self.video_writer = None
                self.show_message("Video record start failed", 2.0)
                return

            self.recording = True
            self.show_message(f"Recording started: {self.record_path.name}", 2.0)
        else:
            self.recording = False
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            self.show_message("Recording stopped", 1.8)

    def save_snapshot(self, scene):
        ts = time.strftime("%Y%m%d_%H%M%S")
        canvas_path = self.save_dir / f"canvas_{ts}.png"
        scene_path = self.save_dir / f"scene_{ts}.png"

        cv2.imwrite(str(canvas_path), self.canvas)
        cv2.imwrite(str(scene_path), scene)

        self.show_message(f"Saved: {scene_path.name}", 2.0)

    # =========================
    # DRAWING / HISTORY
    # =========================
    def smooth_point(self, state, pt):
        state["buffer"].append(pt)
        xs = [p[0] for p in state["buffer"]]
        ys = [p[1] for p in state["buffer"]]
        return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))

    def begin_action(self, state, pt, kind="pen"):
        state["buffer"].clear()
        state["buffer"].append(pt)

        if kind == "pen":
            color = self.current_color
            thickness = self.current_brush
            sig = ("pen", color, thickness)
        else:
            color = (0, 0, 0)
            thickness = max(48, self.current_brush * 6)
            sig = ("erase", thickness)

        action = Action(kind=kind, points=[pt], color=color, thickness=thickness)
        self.actions.append(action)
        state["action"] = action
        state["sig"] = sig

    def stamp_line(self, img, p1, p2, color, radius):
        dist = max(1, int(self.gesture_engine.distance(p1, p2)))
        step = max(1, radius // 3)
        n = max(1, dist // step)

        for i in range(n + 1):
            t = i / n
            x = int(p1[0] + (p2[0] - p1[0]) * t)
            y = int(p1[1] + (p2[1] - p1[1]) * t)
            cv2.circle(img, (x, y), radius, color, -1, cv2.LINE_AA)

    def draw_segment(self, img, action, p1, p2):
        if action.kind == "pen":
            cv2.line(img, p1, p2, action.color, action.thickness, cv2.LINE_AA)
            self.stamp_line(img, p1, p2, action.color, max(1, action.thickness // 2))
        elif action.kind == "erase":
            erase_color = (0, 0, 0)
            cv2.line(img, p1, p2, erase_color, action.thickness, cv2.LINE_AA)
            self.stamp_line(img, p1, p2, erase_color, max(1, action.thickness // 2))

    def extend_action(self, state, pt):
        if state["action"] is None:
            return

        last_pt = state["action"].points[-1]
        if pt == last_pt:
            return

        self.draw_segment(self.canvas, state["action"], last_pt, pt)
        state["action"].points.append(pt)

    def finish_action(self, state):
        if state["action"] is not None:
            if len(state["action"].points) == 1:
                p = state["action"].points[0]
                r = max(2, state["action"].thickness // 2)
                if state["action"].kind == "pen":
                    cv2.circle(self.canvas, p, r, state["action"].color, -1, cv2.LINE_AA)
                else:
                    cv2.circle(self.canvas, p, r, (0, 0, 0), -1, cv2.LINE_AA)

            state["action"] = None
            state["sig"] = None
            state["buffer"].clear()

    def draw_shape_action(self, img, action):
        if len(action.points) < 2:
            return

        p1 = action.points[0]
        p2 = action.points[-1]

        if action.shape_type == "line":
            cv2.line(img, p1, p2, action.color, action.thickness, cv2.LINE_AA)
        elif action.shape_type == "rect":
            cv2.rectangle(img, p1, p2, action.color, action.thickness)
        elif action.shape_type == "circle":
            r = max(1, int(self.gesture_engine.distance(p1, p2)))
            cv2.circle(img, p1, r, action.color, action.thickness, cv2.LINE_AA)

    def draw_shape_preview(self, img, state):
        if not state["shape_active"] or state["shape_start"] is None or state["shape_end"] is None:
            return

        p1 = state["shape_start"]
        p2 = state["shape_end"]
        color = state["shape_color"]
        thickness = state["shape_thickness"]
        shape_type = state["shape_type"]

        if shape_type == "line":
            cv2.line(img, p1, p2, color, thickness, cv2.LINE_AA)
        elif shape_type == "rect":
            cv2.rectangle(img, p1, p2, color, thickness)
        elif shape_type == "circle":
            r = max(1, int(self.gesture_engine.distance(p1, p2)))
            cv2.circle(img, p1, r, color, thickness, cv2.LINE_AA)

    def draw_active_shape_previews(self, img):
        for state in self.hand_states.values():
            self.draw_shape_preview(img, state)

    def start_shape(self, state, pt):
        state["buffer"].clear()
        state["shape_active"] = True
        state["shape_start"] = pt
        state["shape_end"] = pt
        state["shape_color"] = self.current_color
        state["shape_thickness"] = self.current_brush
        state["shape_type"] = self.shape_type

    def update_shape(self, state, pt):
        if not state["shape_active"]:
            self.start_shape(state, pt)
        state["shape_end"] = pt

    def finish_shape(self, state):
        if state["shape_active"] and state["shape_start"] is not None and state["shape_end"] is not None:
            action = Action(
                kind="shape",
                points=[state["shape_start"], state["shape_end"]],
                color=state["shape_color"],
                thickness=state["shape_thickness"],
                shape_type=state["shape_type"],
            )
            self.actions.append(action)
            self.draw_shape_action(self.canvas, action)

        state["shape_active"] = False
        state["shape_start"] = None
        state["shape_end"] = None

    def finish_state(self, state):
        self.finish_action(state)
        self.finish_shape(state)

    def redraw_canvas(self):
        self.canvas[:] = 0
        for action in self.actions:
            if action.kind in ("pen", "erase"):
                pts = action.points
                if len(pts) == 1:
                    p = pts[0]
                    r = max(2, action.thickness // 2)
                    if action.kind == "pen":
                        cv2.circle(self.canvas, p, r, action.color, -1, cv2.LINE_AA)
                    else:
                        cv2.circle(self.canvas, p, r, (0, 0, 0), -1, cv2.LINE_AA)
                else:
                    for i in range(1, len(pts)):
                        self.draw_segment(self.canvas, action, pts[i - 1], pts[i])
            elif action.kind == "shape":
                self.draw_shape_action(self.canvas, action)

    def clear_canvas(self):
        for state in self.hand_states.values():
            state["action"] = None
            state["sig"] = None
            state["buffer"].clear()
            state["shape_active"] = False
            state["shape_start"] = None
            state["shape_end"] = None
            state["fist_start"] = None

        self.actions.clear()
        self.canvas[:] = 0
        self.show_message("Canvas cleared")

    def undo_last(self):
        for state in self.hand_states.values():
            self.finish_state(state)

        if self.actions:
            self.actions.pop()
            self.redraw_canvas()
            self.show_message("Undo done")
        else:
            self.show_message("Nothing to undo")

    # =========================
    # BUTTON ACTIONS
    # =========================
    def execute_button(self, btn):
        if btn.kind == "color":
            self.current_color = btn.value
            self.tool = "pen"
            self.show_message(f"Color: {btn.name}")
        elif btn.kind == "brush":
            self.current_brush = btn.value
            self.show_message(f"Brush: {btn.value}")
        elif btn.kind == "tool":
            self.tool = btn.value
            self.show_message("Pen mode" if btn.value == "pen" else "Eraser mode")
        elif btn.kind == "mode":
            self.global_mode = "shape" if self.global_mode != "shape" else "free"
            self.show_message("Shape mode ON" if self.global_mode == "shape" else "Free mode")
        elif btn.kind == "action":
            if btn.value == "undo":
                self.undo_last()
            elif btn.value == "clear":
                self.clear_canvas()
            elif btn.value == "save":
                self.save_requested = True

    def process_button(self, pt):
        now = time.time()
        if now < self.select_cooldown_until:
            return

        for btn in self.buttons:
            if self.inside(pt, btn.rect):
                self.execute_button(btn)
                self.select_cooldown_until = now + 0.25
                break

    # =========================
    # COMPOSE
    # =========================
    def compose_scene(self, frame):
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        inv = cv2.bitwise_not(mask)

        bg = cv2.bitwise_and(frame, frame, mask=inv)
        fg = cv2.bitwise_and(self.canvas, self.canvas, mask=mask)

        return cv2.add(bg, fg)

    def draw_hand_info(self, frame, lm_list, hand_name, gesture):
        if not lm_list:
            return

        x, y = lm_list[0][1], lm_list[0][2]
        y = max(25, y - 16)

        label = f"{hand_name}:{gesture}"
        w = 155
        cv2.rectangle(frame, (x - 6, y - 18), (x - 6 + w, y + 6), (0, 0, 0), -1)
        cv2.putText(frame, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 2, cv2.LINE_AA)

    # =========================
    # HAND PROCESSING
    # =========================
    def process_hand(self, frame, state, lm_list, fingers, hand_name):
        if not lm_list:
            self.finish_state(state)
            return "NONE"

        now = time.time()
        gesture, pinch_ratio = self.gesture_engine.recognize(fingers, lm_list)

        ix, iy = lm_list[8][1], lm_list[8][2]

        # Fist hold = clear
        if gesture == "FIST":
            self.finish_state(state)

            if state["fist_start"] is None:
                state["fist_start"] = now
            else:
                held = now - state["fist_start"]
                if held >= self.clear_hold_seconds:
                    self.clear_canvas()
                    state["fist_start"] = None
                    self.show_message("Canvas cleared by fist gesture", 1.5)

            self.mode = "FIST"
            self.draw_hand_info(frame, lm_list, hand_name, gesture)
            return gesture

        state["fist_start"] = None

        # Pinch = brush control
        if gesture == "PINCH":
            self.finish_state(state)

            new_brush = int(np.interp(np.clip(pinch_ratio, 0.12, 0.85), [0.12, 0.85], [4, 36]))
            new_brush = int(np.clip(new_brush, 2, 50))

            if abs(new_brush - self.current_brush) >= 1:
                self.current_brush = int(0.75 * self.current_brush + 0.25 * new_brush)
                if now > self.pinch_msg_until:
                    self.show_message(f"Pinch brush: {self.current_brush}", 0.8)
                    self.pinch_msg_until = now + 0.20

            self.mode = "PINCH"
            self.draw_hand_info(frame, lm_list, hand_name, gesture)
            return gesture

        # Open palm = pause
        if gesture == "Hand":
            self.finish_state(state)
            self.mode = "PAUSE"
            self.draw_hand_info(frame, lm_list, hand_name, gesture)
            return gesture

        # Two fingers = select toolbar
        if gesture == "TWO":
            self.finish_state(state)
            self.mode = "SELECT"
            self.process_button((ix, iy))
            self.draw_hand_info(frame, lm_list, hand_name, gesture)
            return gesture

        # Three fingers = erase
        if gesture == "THREE":
            self.finish_shape(state)
            self.tool = "eraser"
            self.mode = "ERASE"

            if iy > TOP_H + 5:
                pt = self.smooth_point(state, (ix, iy))
                desired_sig = ("erase", max(48, self.current_brush * 6))

                if state["action"] is None or state["sig"] != desired_sig:
                    self.finish_action(state)
                    self.begin_action(state, pt, kind="erase")
                else:
                    self.extend_action(state, pt)
            else:
                self.finish_action(state)

            self.draw_hand_info(frame, lm_list, hand_name, gesture)
            return gesture

        # One finger = ALWAYS draw
        if gesture == "POINT":
            if self.global_mode == "shape":
                self.mode = "SHAPE"
                self.finish_action(state)

                if iy > TOP_H + 5:
                    pt = self.smooth_point(state, (ix, iy))
                    self.update_shape(state, pt)
                else:
                    self.finish_shape(state)
            else:
                # IMPORTANT:
                # When 1 finger appears, auto switch back to pen/draw mode
                self.tool = "pen"

                if iy > TOP_H + 5:
                    pt = self.smooth_point(state, (ix, iy))
                    desired_sig = ("pen", self.current_color, self.current_brush)

                    if state["action"] is None or state["sig"] != desired_sig:
                        self.finish_action(state)
                        self.begin_action(state, pt, kind="pen")
                    else:
                        self.extend_action(state, pt)

                    self.mode = "DRAW"
                else:
                    self.finish_action(state)
                    self.finish_shape(state)

            self.draw_hand_info(frame, lm_list, hand_name, gesture)
            return gesture

        # Any other custom gesture
        self.finish_state(state)
        self.mode = gesture
        self.draw_hand_info(frame, lm_list, hand_name, gesture)
        return gesture

    # =========================
    # MAIN LOOP
    # =========================
    def run(self):
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, WIDTH, HEIGHT)

        while True:
            ok, frame = self.cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)

            results = self.tracker.find_hands(frame)
            hands = results.multi_hand_landmarks if results and results.multi_hand_landmarks else []

            seen_keys = set()
            active_modes = []

            for idx, _ in enumerate(hands):
                hand_name = self.tracker.get_hand_label(results, idx) or f"H{idx}"
                key = f"{hand_name}_{idx}"
                seen_keys.add(key)

                state = self.hand_states.setdefault(key, self.new_hand_state())
                state["last_seen"] = time.time()

                lm_list = self.tracker.get_landmarks(frame, results, hand_index=idx, draw=True)
                fingers = self.tracker.fingers_up(lm_list, hand_name)

                gesture = self.process_hand(frame, state, lm_list, fingers, hand_name)
                active_modes.append(f"{hand_name}:{gesture}")

            # cleanup lost hands
            for key in list(self.hand_states.keys()):
                if key not in seen_keys:
                    self.finish_state(self.hand_states[key])
                    del self.hand_states[key]

            # FPS
            self.fps_frame_count += 1
            now = time.time()
            if now - self.fps_last_time >= 1.0:
                self.fps = self.fps_frame_count / (now - self.fps_last_time)
                self.fps_frame_count = 0
                self.fps_last_time = now

            self.mode = " | ".join(active_modes) if active_modes else "IDLE"

            # compose scene
            scene = self.compose_scene(frame)

            # save snapshot if requested
            if self.save_requested:
                self.save_snapshot(scene.copy())
                self.save_requested = False

            # display
            display = scene.copy()
            self.draw_active_shape_previews(display)
            self.draw_toolbar(display)
            self.draw_status(display, len(hands))
            self.draw_help(display)

            # cursor only below toolbar
            if hands:
                for idx, _ in enumerate(hands):
                    lm_list = self.tracker.get_landmarks(frame, results, hand_index=idx, draw=False)
                    if lm_list:
                        ix, iy = lm_list[8][1], lm_list[8][2]
                        if iy > TOP_H + 6:
                            cursor_color = (0, 255, 255) if self.mode == "SELECT" else (0, 255, 0) if self.mode == "DRAW" else (255, 255, 255)
                            cv2.circle(display, (ix, iy), 6, cursor_color, -1, cv2.LINE_AA)

            cv2.imshow(WINDOW_NAME, display)

            # record frame
            if self.recording and self.video_writer is not None:
                self.video_writer.write(display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("z"):
                self.undo_last()
            elif key == ord("c"):
                self.clear_canvas()
            elif key == ord("s"):
                self.save_requested = True
            elif key == ord("r"):
                self.toggle_recording()
            elif key == ord("h"):
                self.help_visible = not self.help_visible
                self.show_message("Help ON" if self.help_visible else "Help OFF", 1.0)
            elif key == ord("p"):
                self.tool = "pen"
                self.show_message("Pen mode")
            elif key == ord("e"):
                self.tool = "eraser"
                self.show_message("Eraser mode")
            elif key == ord("m"):
                self.global_mode = "shape" if self.global_mode != "shape" else "free"
                self.show_message("Shape mode ON" if self.global_mode == "shape" else "Free mode")
            elif key == ord("l"):
                self.shape_type = "line"
                self.show_message("Shape: line")
            elif key == ord("b"):
                self.shape_type = "rect"
                self.show_message("Shape: rect")
            elif key == ord("o"):
                self.shape_type = "circle"
                self.show_message("Shape: circle")
            elif key == ord("1"):
                self.current_brush = BRUSH_PRESETS[0]
                self.show_message(f"Brush: {self.current_brush}")
            elif key == ord("2"):
                self.current_brush = BRUSH_PRESETS[1]
                self.show_message(f"Brush: {self.current_brush}")
            elif key == ord("3"):
                self.current_brush = BRUSH_PRESETS[2]
                self.show_message(f"Brush: {self.current_brush}")
            elif key == ord("4"):
                self.current_brush = BRUSH_PRESETS[3]
                self.show_message(f"Brush: {self.current_brush}")
            elif key == ord("5"):
                self.current_brush = BRUSH_PRESETS[4]
                self.show_message(f"Brush: {self.current_brush}")

        self.cap.release()
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = VirtualPainter()
    app.run()