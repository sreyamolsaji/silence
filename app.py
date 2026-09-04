import time
import threading
import queue
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from scipy.signal import resample_poly

try:
    import sounddevice as sd
except (ImportError, OSError):
    sd = None

import mediapipe as mp
from silero_vad import load_silero_vad


# ============================================================
# SETTINGS
# ============================================================

AUDIO_DEVICE = 1

CAPTURE_RATE = 44100
VAD_RATE = 16000
AUDIO_BLOCK = 4410

CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
CAMERA_FPS = 30

SPEECH_ON = 0.55
SPEECH_OFF = 0.35

VOICE_CONFIRM_FRAMES = 2
VOICE_RELEASE_FRAMES = 4

MAX_HISTORY = 240

DEFAULT_DURATION = 30


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Silence Compressor™",
    page_icon="🤐",
    layout="wide"
)


# ============================================================
# UI STYLE
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
        radial-gradient(
            circle at top left,
            #17172a 0%,
            #0b0b12 45%,
            #07070b 100%
        );
    }

    .main-title {
        font-size: 3.2rem;
        font-weight: 900;
        letter-spacing: -1px;
        margin-bottom: 0;
    }

    .subtitle {
        color: #9ca3af;
        font-size: 1.05rem;
        margin-top: 5px;
        margin-bottom: 25px;
    }

    div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 14px;
    }

    .score-box {
        text-align: center;
        padding: 28px;
        margin-top: 10px;
        margin-bottom: 20px;
        border-radius: 20px;
        background: rgba(255,255,255,0.035);
        border: 1px solid rgba(255,255,255,0.08);
    }

    .score-number {
        font-size: 4.5rem;
        font-weight: 900;
        line-height: 1;
    }

    .score-label {
        font-size: 1.4rem;
        font-weight: 800;
        margin-top: 12px;
    }

    .timeline-title {
        font-size: 1.1rem;
        font-weight: 800;
        margin-top: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SCORING
# ============================================================

def score_silence(seconds):

    if seconds < 1:
        return 0

    elif seconds < 2:
        return 10

    elif seconds < 3:
        return 30

    elif seconds < 5:
        return 60

    elif seconds < 8:
        return 85

    elif seconds < 12:
        return 95

    else:
        return 100


def awkward_label(score):

    if score < 20:
        return "COMFORTABLE"

    elif score < 40:
        return "SLIGHTLY AWKWARD"

    elif score < 60:
        return "GETTING AWKWARD"

    elif score < 80:
        return "VERY AWKWARD"

    elif score < 90:
        return "🚨 ELITE AWKWARDNESS"

    else:
        return "👑 LEGENDARY AWKWARDNESS"


def get_blendshape_dict(face_blendshapes):

    result = {}

    for category in face_blendshapes:
        result[category.category_name] = category.score

    return result


def value(dictionary, name):

    return float(
        dictionary.get(name, 0.0)
    )


# ============================================================
# ENGINE
# ============================================================

class MultimodalEngine:

    def __init__(self, duration=30):

        self.duration = duration

        self.lock = threading.Lock()

        self.running = False
        self.finished = False

        self.start_time = None

        self.error_message = ""

        # ====================================================
        # AUDIO
        # ====================================================

        self.audio_queue = queue.Queue(
            maxsize=50
        )

        self.audio_state = "WAITING"

        self.speech_probability = 0.0
        self.audio_rms = 0.0
        self.noise_floor = 0.005

        self.speech_active = False

        self.speech_confirm_counter = 0
        self.speech_release_counter = 0

        self.current_silence_start = None
        self.current_silence = 0.0

        self.longest_silence = 0.0
        self.silence_events = 0

        self.speech_samples = 0
        self.noise_samples = 0

        # ====================================================
        # CAMERA
        # ====================================================

        self.latest_frame = None

        self.people_count = 0

        self.camera_fps = 0.0

        self.social_score = 0

        self.mutual_interaction = False

        self.face_information = []

        # ====================================================
        # SCORE HISTORY
        # ====================================================

        self.time_history = deque(
            maxlen=MAX_HISTORY
        )

        self.score_history = deque(
            maxlen=MAX_HISTORY
        )

        # ====================================================
        # PEAK
        # ====================================================

        self.peak_score = 0
        self.peak_time = 0.0
        self.peak_label = "COMFORTABLE"

        self.current_label = "COMFORTABLE"

        # ====================================================
        # THREADS
        # ====================================================

        self.audio_thread = None
        self.camera_thread = None

        self.audio_stream = None

        self.vad_model = None

        # ====================================================
        # FINAL
        # ====================================================

        self.final_score = 0
        self.final_label = ""

    # ========================================================
    # AUDIO CALLBACK
    # ========================================================

    def audio_callback(
        self,
        indata,
        frames,
        time_info,
        status
    ):

        try:

            audio = indata[:, 0].copy()

            try:

                self.audio_queue.put_nowait(
                    audio
                )

            except queue.Full:

                pass

        except Exception as e:

            self.error_message = (
                f"Audio callback error: {e}"
            )

    # ========================================================
    # AUDIO WORKER
    # ========================================================

    def audio_worker(self):

        try:

            self.vad_model = load_silero_vad()

            vad_buffer = np.zeros(
                0,
                dtype=np.float32
            )

            while self.running:

                try:

                    audio = (
                        self.audio_queue.get(
                            timeout=0.2
                        )
                    )

                except queue.Empty:

                    continue

                if audio is None:
                    break

                # --------------------------------------------
                # RMS
                # --------------------------------------------

                rms = float(
                    np.sqrt(
                        np.mean(
                            audio ** 2
                        )
                    ) + 1e-10
                )

                # --------------------------------------------
                # RESAMPLE
                # --------------------------------------------

                audio_16k = (
                    resample_poly(
                        audio,
                        VAD_RATE,
                        CAPTURE_RATE
                    ).astype(
                        np.float32
                    )
                )

                vad_buffer = np.concatenate(
                    [
                        vad_buffer,
                        audio_16k
                    ]
                )

                # --------------------------------------------
                # EXACT 512 SAMPLE CHUNKS
                # --------------------------------------------

                while len(vad_buffer) >= 512:

                    chunk = vad_buffer[:512]

                    vad_buffer = (
                        vad_buffer[512:]
                    )

                    try:

                        import torch

                        tensor_chunk = (
                            torch.from_numpy(
                                chunk.astype(
                                    np.float32
                                )
                            )
                        )

                        probability = float(
                            self.vad_model(
                                tensor_chunk,
                                VAD_RATE
                            ).item()
                        )

                    except Exception:

                        probability = 0.0

                    with self.lock:

                        self.speech_probability = (
                            probability
                        )

                        self.audio_rms = rms

                        # ------------------------------------
                        # NOISE FLOOR
                        # ------------------------------------

                        if not self.speech_active:

                            self.noise_floor = (
                                0.98
                                * self.noise_floor
                                + 0.02 * rms
                            )

                        rms_gate = max(
                            self.noise_floor * 1.5,
                            0.0015
                        )

                        possible_speech = (
                            probability >= SPEECH_ON
                            and rms >= rms_gate
                        )

                        possible_non_speech = (
                            probability <= SPEECH_OFF
                        )

                        # ------------------------------------
                        # HYSTERESIS
                        # ------------------------------------

                        if not self.speech_active:

                            if possible_speech:

                                self.speech_confirm_counter += 1

                            else:

                                self.speech_confirm_counter = 0

                            if (
                                self.speech_confirm_counter
                                >= VOICE_CONFIRM_FRAMES
                            ):

                                self.speech_active = True

                                self.speech_release_counter = 0

                        else:

                            if possible_non_speech:

                                self.speech_release_counter += 1

                            else:

                                self.speech_release_counter = 0

                            if (
                                self.speech_release_counter
                                >= VOICE_RELEASE_FRAMES
                            ):

                                self.speech_active = False

                                self.speech_confirm_counter = 0

                        # ------------------------------------
                        # SPEECH
                        # ------------------------------------

                        if self.speech_active:

                            self.audio_state = (
                                "🗣️ SPEECH"
                            )

                            self.speech_samples += 1

                            if (
                                self.current_silence_start
                                is not None
                            ):

                                duration = (
                                    time.monotonic()
                                    - self.current_silence_start
                                )

                                if duration >= 0.7:

                                    self.silence_events += 1

                                if (
                                    duration
                                    > self.longest_silence
                                ):

                                    self.longest_silence = (
                                        duration
                                    )

                                self.current_silence_start = None

                                self.current_silence = 0.0

                        # ------------------------------------
                        # NOISE / TRUE SILENCE
                        # ------------------------------------

                        else:

                            if rms > max(
                                self.noise_floor * 2.0,
                                0.003
                            ):

                                self.audio_state = (
                                    "🔊 NOISE"
                                )

                                self.noise_samples += 1

                            else:

                                self.audio_state = (
                                    "🔇 TRUE SILENCE"
                                )

                                if (
                                    self.current_silence_start
                                    is None
                                ):

                                    self.current_silence_start = (
                                        time.monotonic()
                                    )

                                self.current_silence = (
                                    time.monotonic()
                                    - self.current_silence_start
                                )

                                if (
                                    self.current_silence
                                    > self.longest_silence
                                ):

                                    self.longest_silence = (
                                        self.current_silence
                                    )

        except Exception as e:

            self.error_message = (
                f"Audio engine error: {e}"
            )

    # ========================================================
    # FACING
    # ========================================================

    def estimate_facing(
        self,
        landmarks
    ):

        try:

            nose = landmarks[1]

            left_eye = landmarks[33]
            right_eye = landmarks[263]

            eye_center_x = (
                left_eye.x
                + right_eye.x
            ) / 2

            difference = (
                nose.x
                - eye_center_x
            )

            if difference > 0.025:

                return "LEFT"

            elif difference < -0.025:

                return "RIGHT"

            return "CENTER"

        except Exception:

            return "CENTER"

    # ========================================================
    # GAZE
    # ========================================================

    def estimate_gaze(
        self,
        blend
    ):

        left_in = value(
            blend,
            "eyeLookInLeft"
        )

        left_out = value(
            blend,
            "eyeLookOutLeft"
        )

        right_in = value(
            blend,
            "eyeLookInRight"
        )

        right_out = value(
            blend,
            "eyeLookOutRight"
        )

        left_down = value(
            blend,
            "eyeLookDownLeft"
        )

        right_down = value(
            blend,
            "eyeLookDownRight"
        )

        down = (
            left_down
            + right_down
        ) / 2

        left_score = (
            left_out
            + right_in
        ) / 2

        right_score = (
            left_in
            + right_out
        ) / 2

        if down > 0.35:

            return "DOWN"

        if left_score > (
            right_score + 0.12
        ):

            return "LEFT"

        if right_score > (
            left_score + 0.12
        ):

            return "RIGHT"

        return "CENTER"

    # ========================================================
    # EXPRESSION
    # ========================================================

    def estimate_expression(
        self,
        blend
    ):

        smile = (
            value(
                blend,
                "mouthSmileLeft"
            )
            + value(
                blend,
                "mouthSmileRight"
            )
        ) / 2

        frown = (
            value(
                blend,
                "mouthFrownLeft"
            )
            + value(
                blend,
                "mouthFrownRight"
            )
        ) / 2

        brow_up = (
            value(
                blend,
                "browInnerUp"
            )
            + value(
                blend,
                "browOuterUpLeft"
            )
            + value(
                blend,
                "browOuterUpRight"
            )
        ) / 3

        brow_down = (
            value(
                blend,
                "browDownLeft"
            )
            + value(
                blend,
                "browDownRight"
            )
        ) / 2

        jaw_open = value(
            blend,
            "jawOpen"
        )

        if smile > 0.45:
            return "HAPPY"

        if frown > 0.35:
            return "SAD"

        if (
            jaw_open > 0.45
            and brow_up > 0.25
        ):
            return "SURPRISED"

        if (
            brow_down > 0.4
            and frown > 0.2
        ):
            return "ANGRY"

        return "NEUTRAL"

    # ========================================================
    # SOCIAL SCORE
    # ========================================================

    def calculate_social_score(
        self,
        gaze,
        expression,
        facing
    ):

        score = 50

        if gaze == "DOWN":

            score += 20

        elif gaze in [
            "LEFT",
            "RIGHT"
        ]:

            score += 8

        elif gaze == "CENTER":

            score -= 5

        if expression == "HAPPY":

            score -= 8

        if expression == "SAD":

            score += 8

        if expression == "SURPRISED":

            score += 5

        if facing == "CENTER":

            score -= 3

        return int(
            np.clip(
                score,
                0,
                100
            )
        )

    # ========================================================
    # CAMERA
    # ========================================================

    def camera_worker(self):

        try:

            cap = cv2.VideoCapture(
                0,
                cv2.CAP_DSHOW
            )

            if not cap.isOpened():

                cap.release()

                cap = cv2.VideoCapture(0)

            if not cap.isOpened():

                raise RuntimeError(
                    "Could not open webcam."
                )

            cap.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                CAMERA_WIDTH
            )

            cap.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                CAMERA_HEIGHT
            )

            cap.set(
                cv2.CAP_PROP_FPS,
                CAMERA_FPS
            )

            try:

                cap.set(
                    cv2.CAP_PROP_FOURCC,
                    cv2.VideoWriter_fourcc(
                        *"MJPG"
                    )
                )

            except Exception:

                pass

            # --------------------------------------------
            # MEDIAPIPE
            # --------------------------------------------

            BaseOptions = (
                mp.tasks.BaseOptions
            )

            RunningMode = (
                mp.tasks.vision.RunningMode
            )

            options = (
                mp.tasks.vision.FaceLandmarkerOptions(
                    base_options=BaseOptions(
                        model_asset_path=
                        str(
                            Path(__file__).with_name(
                                "face_landmarker.task"
                            )
                        )
                    ),
                    running_mode=RunningMode.VIDEO,
                    num_faces=4,
                    min_face_detection_confidence=0.5,
                    min_face_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                    output_face_blendshapes=True,
                    output_facial_transformation_matrixes=False
                )
            )

            landmarker = (
                mp.tasks.vision.FaceLandmarker
                .create_from_options(
                    options
                )
            )

            frame_count = 0

            fps_start = time.monotonic()

            last_timestamp = 0

            while self.running:

                ok, frame = cap.read()

                if not ok:

                    continue

                display_frame = (
                    frame.copy()
                )

                rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=(
                        mp.ImageFormat.SRGB
                    ),
                    data=rgb
                )

                timestamp = int(
                    time.monotonic()
                    * 1000
                )

                if timestamp <= last_timestamp:

                    timestamp = (
                        last_timestamp + 1
                    )

                last_timestamp = timestamp

                result = (
                    landmarker.detect_for_video(
                        mp_image,
                        timestamp
                    )
                )

                face_info = []

                # ----------------------------------------
                # FACE PROCESSING
                # ----------------------------------------

                if result.face_landmarks:

                    for i, landmarks in enumerate(
                        result.face_landmarks
                    ):

                        blend = {}

                        if result.face_blendshapes:

                            blend = (
                                get_blendshape_dict(
                                    result.face_blendshapes[i]
                                )
                            )

                        facing = (
                            self.estimate_facing(
                                landmarks
                            )
                        )

                        gaze = (
                            self.estimate_gaze(
                                blend
                            )
                        )

                        expression = (
                            self.estimate_expression(
                                blend
                            )
                        )

                        social_score = (
                            self.calculate_social_score(
                                gaze,
                                expression,
                                facing
                            )
                        )

                        nose = landmarks[1]

                        x = int(
                            nose.x
                            * frame.shape[1]
                        )

                        y = int(
                            nose.y
                            * frame.shape[0]
                        )

                        face_info.append(
                            {
                                "center": (x, y),
                                "gaze": gaze,
                                "facing": facing,
                                "expression": expression,
                                "social_score": social_score
                            }
                        )

                        # --------------------------------
                        # FACE BOX
                        # --------------------------------

                        xs = [
                            int(
                                p.x
                                * frame.shape[1]
                            )
                            for p in landmarks
                        ]

                        ys = [
                            int(
                                p.y
                                * frame.shape[0]
                            )
                            for p in landmarks
                        ]

                        x1 = max(
                            0,
                            min(xs)
                        )

                        y1 = max(
                            0,
                            min(ys)
                        )

                        x2 = min(
                            frame.shape[1] - 1,
                            max(xs)
                        )

                        y2 = min(
                            frame.shape[0] - 1,
                            max(ys)
                        )

                        cv2.rectangle(
                            display_frame,
                            (x1, y1),
                            (x2, y2),
                            (255, 0, 255),
                            2
                        )

                        label = (
                            f"Person {i+1} | "
                            f"Gaze: {gaze} | "
                            f"{expression}"
                        )

                        cv2.putText(
                            display_frame,
                            label,
                            (
                                x1,
                                max(
                                    25,
                                    y1 - 10
                                )
                            ),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 0, 255),
                            2
                        )

                # ----------------------------------------
                # MUTUAL INTERACTION
                # ----------------------------------------

                mutual = False

                if len(face_info) >= 2:

                    sorted_faces = sorted(
                        face_info,
                        key=lambda f:
                        f["center"][0]
                    )

                    left_person = (
                        sorted_faces[0]
                    )

                    right_person = (
                        sorted_faces[1]
                    )

                    if (
                        left_person["facing"]
                        == "RIGHT"
                        and
                        right_person["facing"]
                        == "LEFT"
                    ):

                        mutual = True

                        cv2.putText(
                            display_frame,
                            "MUTUAL INTERACTION: YES",
                            (20, 95),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 0, 255),
                            2
                        )

                # ----------------------------------------
                # SOCIAL SCORE
                # ----------------------------------------

                if face_info:

                    social_score = int(
                        np.clip(
                            np.mean(
                                [
                                    f[
                                        "social_score"
                                    ]
                                    for f in face_info
                                ]
                            ),
                            0,
                            100
                        )
                    )

                else:

                    social_score = 0

                # ----------------------------------------
                # HUD
                # ----------------------------------------

                cv2.putText(
                    display_frame,
                    f"PEOPLE: {len(face_info)}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 255),
                    2
                )

                cv2.putText(
                    display_frame,
                    f"FPS: {self.camera_fps:.1f}",
                    (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 0, 255),
                    2
                )

                cv2.putText(
                    display_frame,
                    "SOCIAL CUE ENGINE: ACTIVE",
                    (20, 125),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 0, 255),
                    2
                )

                # ----------------------------------------
                # SHARED STATE
                # ----------------------------------------

                with self.lock:

                    self.latest_frame = (
                        display_frame
                    )

                    self.people_count = (
                        len(face_info)
                    )

                    self.face_information = (
                        face_info
                    )

                    self.social_score = (
                        social_score
                    )

                    self.mutual_interaction = (
                        mutual
                    )

                # ----------------------------------------
                # FPS
                # ----------------------------------------

                frame_count += 1

                elapsed = (
                    time.monotonic()
                    - fps_start
                )

                if elapsed >= 1:

                    self.camera_fps = (
                        frame_count
                        / elapsed
                    )

                    frame_count = 0

                    fps_start = (
                        time.monotonic()
                    )

            landmarker.close()

            cap.release()

        except Exception as e:

            self.error_message = (
                f"Camera engine error: {e}"
            )

    # ========================================================
    # START
    # ========================================================

    def start(self):

        if self.running:

            return

        self.running = True

        self.finished = False

        self.start_time = (
            time.monotonic()
        )

        self.final_score = 0

        self.final_label = ""

        self.error_message = ""

        # --------------------------------------------
        # RESET AUDIO
        # --------------------------------------------

        self.audio_state = "WAITING"

        self.speech_probability = 0.0

        self.audio_rms = 0.0

        self.noise_floor = 0.005

        self.speech_active = False

        self.speech_confirm_counter = 0

        self.speech_release_counter = 0

        self.current_silence_start = None

        self.current_silence = 0.0

        self.longest_silence = 0.0

        self.silence_events = 0

        self.speech_samples = 0

        self.noise_samples = 0

        # --------------------------------------------
        # RESET VISION
        # --------------------------------------------

        self.latest_frame = None

        self.people_count = 0

        self.social_score = 0

        self.mutual_interaction = False

        self.face_information = []

        # --------------------------------------------
        # RESET TIMELINE
        # --------------------------------------------

        self.time_history.clear()

        self.score_history.clear()

        self.peak_score = 0

        self.peak_time = 0.0

        self.peak_label = "COMFORTABLE"

        self.current_label = "COMFORTABLE"

        # --------------------------------------------
        # THREADS
        # --------------------------------------------

        self.audio_thread = threading.Thread(
            target=self.audio_worker,
            daemon=True
        )

        self.camera_thread = threading.Thread(
            target=self.camera_worker,
            daemon=True
        )

        # --------------------------------------------
        # MICROPHONE
        # --------------------------------------------

        try:

            if sd is None:
                raise RuntimeError(
                    "Audio input is unavailable in this deployment."
                )

            self.audio_stream = (
                sd.InputStream(
                    device=AUDIO_DEVICE,
                    samplerate=CAPTURE_RATE,
                    channels=1,
                    dtype="float32",
                    blocksize=AUDIO_BLOCK,
                    callback=self.audio_callback
                )
            )

            self.audio_stream.start()

        except Exception as e:

            self.running = False

            self.error_message = (
                f"Could not start microphone: {e}"
            )

            return

        self.audio_thread.start()

        self.camera_thread.start()

    # ========================================================
    # STOP
    # ========================================================

    def stop(self):

        if not self.running:

            return

        self.running = False

        try:

            if self.audio_stream is not None:

                self.audio_stream.stop()

                self.audio_stream.close()

        except Exception:

            pass

        try:

            self.audio_queue.put_nowait(
                None
            )

        except Exception:

            pass

        time.sleep(0.3)

        self.finalize()

    # ========================================================
    # FINALIZE
    # ========================================================

    def finalize(self):

        if self.finished:

            return

        with self.lock:

            # --------------------------------------------
            # CURRENT SILENCE
            # --------------------------------------------

            if (
                self.current_silence_start
                is not None
            ):

                duration = (
                    time.monotonic()
                    - self.current_silence_start
                )

                if (
                    duration
                    > self.longest_silence
                ):

                    self.longest_silence = duration

                if duration >= 0.7:

                    self.silence_events += 1

            # --------------------------------------------
            # AUDIO SCORE
            # --------------------------------------------

            audio_score = score_silence(
                self.longest_silence
            )

            audio_score += min(
                self.silence_events * 4,
                20
            )

            audio_score = int(
                np.clip(
                    audio_score,
                    0,
                    100
                )
            )

            # --------------------------------------------
            # VISION
            # --------------------------------------------

            vision_score = (
                self.social_score
            )

            # --------------------------------------------
            # FUSION
            # --------------------------------------------

            if self.people_count > 0:

                final_score = (
                    0.70 * audio_score
                    + 0.30 * vision_score
                )

            else:

                final_score = audio_score

            self.final_score = int(
                np.clip(
                    final_score,
                    0,
                    100
                )
            )

            self.final_label = (
                awkward_label(
                    self.final_score
                )
            )

            # Make sure final score is recorded
            elapsed = (
                time.monotonic()
                - self.start_time
            )

            self.time_history.append(
                elapsed
            )

            self.score_history.append(
                self.final_score
            )

            if (
                self.final_score
                >= self.peak_score
            ):

                self.peak_score = (
                    self.final_score
                )

                self.peak_time = elapsed

                self.peak_label = (
                    self.final_label
                )

            self.finished = True


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">'
    '🤐 Silence Compressor™'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Real-Time Multimodal Awkwardness Analyzer'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SESSION
# ============================================================

if "engine" not in st.session_state:

    st.session_state.engine = None


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ Experiment Settings"
    )

    duration = st.slider(
        "Analysis duration",
        min_value=10,
        max_value=120,
        value=DEFAULT_DURATION,
        step=5
    )

    st.divider()

    st.write(
        f"**Microphone:** Device {AUDIO_DEVICE}"
    )

    st.write(
        f"**Audio capture:** "
        f"{CAPTURE_RATE} Hz"
    )

    st.write(
        f"**VAD:** "
        f"{VAD_RATE} Hz"
    )

    st.write(
        f"**Camera:** "
        f"{CAMERA_WIDTH} × "
        f"{CAMERA_HEIGHT}"
    )

    st.divider()

    st.info(
        "The audio and camera engines "
        "run continuously in background "
        "workers."
    )


# ============================================================
# BUTTONS
# ============================================================

col1, col2 = st.columns(2)

with col1:

    start_clicked = st.button(
        "▶️ START ANALYSIS",
        use_container_width=True,
        type="primary"
    )

with col2:

    stop_clicked = st.button(
        "⏹️ STOP",
        use_container_width=True
    )


# ============================================================
# START
# ============================================================

if start_clicked:

    if (
        st.session_state.engine is None
        or st.session_state.engine.finished
    ):

        st.session_state.engine = (
            MultimodalEngine(
                duration=duration
            )
        )

    st.session_state.engine.start()


# ============================================================
# STOP
# ============================================================

if stop_clicked:

    if st.session_state.engine is not None:

        st.session_state.engine.stop()


# ============================================================
# LIVE DASHBOARD
# ============================================================

@st.fragment(run_every=0.25)
def live_dashboard():

    engine = st.session_state.engine

    if engine is None:

        st.info(
            "Press **▶️ START ANALYSIS** "
            "to begin."
        )

        return

    # ========================================================
    # FINISH CHECK
    # ========================================================

    if (
        engine.running
        and engine.start_time is not None
        and
        time.monotonic()
        - engine.start_time
        >= engine.duration
    ):

        engine.running = False

        try:

            if engine.audio_stream is not None:

                engine.audio_stream.stop()

                engine.audio_stream.close()

        except Exception:

            pass

        engine.finalize()

    # ========================================================
    # STATUS
    # ========================================================

    if engine.running:

        elapsed = (
            time.monotonic()
            - engine.start_time
        )

        remaining = max(
            0,
            engine.duration
            - elapsed
        )

        st.success(
            f"🟢 SYSTEM ACTIVE  •  "
            f"{remaining:.1f}s remaining"
        )

    elif engine.finished:

        st.warning(
            "⏹️ Analysis complete."
        )

    # ========================================================
    # ERROR
    # ========================================================

    if engine.error_message:

        st.error(
            engine.error_message
        )

    # ========================================================
    # CAMERA + AUDIO
    # ========================================================

    camera_col, audio_col = st.columns(
        [1.35, 1]
    )

    # ========================================================
    # CAMERA
    # ========================================================

    with camera_col:

        st.subheader(
            "📷 Vision Analysis"
        )

        with engine.lock:

            frame = engine.latest_frame

            people = engine.people_count

            fps = engine.camera_fps

            social_score = (
                engine.social_score
            )

            mutual = (
                engine.mutual_interaction
            )

            face_info = list(
                engine.face_information
            )

        if frame is not None:

            st.image(
                frame,
                channels="BGR",
                use_container_width=True
            )

        else:

            st.info(
                "Waiting for webcam..."
            )

        m1, m2, m3 = st.columns(3)

        with m1:

            st.metric(
                "People",
                people
            )

        with m2:

            st.metric(
                "Camera FPS",
                f"{fps:.1f}"
            )

        with m3:

            st.metric(
                "Social Cue",
                f"{social_score}/100"
            )

        if mutual:

            st.success(
                "🤝 Mutual interaction detected"
            )

        if face_info:

            st.write(
                "### 👥 Participants"
            )

            for i, person in enumerate(
                face_info
            ):

                st.write(
                    f"**Person {i+1}:** "
                    f"Gaze `{person['gaze']}` · "
                    f"Facing `{person['facing']}` · "
                    f"Expression "
                    f"`{person['expression']}` · "
                    f"Social cue "
                    f"`{person['social_score']}/100`"
                )

    # ========================================================
    # AUDIO
    # ========================================================

    with audio_col:

        st.subheader(
            "🎙️ Audio Analysis"
        )

        with engine.lock:

            audio_state = (
                engine.audio_state
            )

            probability = (
                engine.speech_probability
            )

            rms = engine.audio_rms

            noise_floor = (
                engine.noise_floor
            )

            silence = (
                engine.current_silence
            )

            longest = (
                engine.longest_silence
            )

            events = (
                engine.silence_events
            )

            speech_samples = (
                engine.speech_samples
            )

            noise_samples = (
                engine.noise_samples
            )

        st.metric(
            "Audio State",
            audio_state
        )

        st.progress(
            min(
                max(
                    probability,
                    0.0
                ),
                1.0
            ),
            text=(
                f"Speech probability: "
                f"{probability:.2f}"
            )
        )

        a1, a2 = st.columns(2)

        with a1:

            st.metric(
                "Current silence",
                f"{silence:.2f}s"
            )

        with a2:

            st.metric(
                "Longest silence",
                f"{longest:.2f}s"
            )

        a3, a4 = st.columns(2)

        with a3:

            st.metric(
                "Silence events",
                events
            )

        with a4:

            st.metric(
                "Noise floor",
                f"{noise_floor:.4f}"
            )

        st.write(
            f"Microphone RMS: "
            f"`{rms:.5f}`"
        )

        st.write(
            f"Speech samples: "
            f"`{speech_samples}`"
        )

        st.write(
            f"Noise samples ignored: "
            f"`{noise_samples}`"
        )

        st.caption(
            "Background noise is not counted "
            "as speech and does not reset "
            "the silence timer."
        )

    # ========================================================
    # LIVE SCORE CALCULATION
    # ========================================================

    with engine.lock:

        longest = (
            engine.longest_silence
        )

        events = (
            engine.silence_events
        )

        vision = (
            engine.social_score
        )

        people = (
            engine.people_count
        )

    audio_live = score_silence(
        longest
    )

    audio_live += min(
        events * 4,
        20
    )

    audio_live = int(
        np.clip(
            audio_live,
            0,
            100
        )
    )

    if people > 0:

        live_score = int(
            np.clip(
                0.70 * audio_live
                + 0.30 * vision,
                0,
                100
            )
        )

    else:

        live_score = audio_live

    # ========================================================
    # RECORD TIMELINE
    # ========================================================

    if engine.running:

        elapsed_time = (
            time.monotonic()
            - engine.start_time
        )

        current_label = (
            awkward_label(
                live_score
            )
        )

        with engine.lock:

            engine.time_history.append(
                elapsed_time
            )

            engine.score_history.append(
                live_score
            )

            engine.current_label = (
                current_label
            )

            if live_score > engine.peak_score:

                engine.peak_score = (
                    live_score
                )

                engine.peak_time = (
                    elapsed_time
                )

                engine.peak_label = (
                    current_label
                )

    # ========================================================
    # LIVE SCORE
    # ========================================================

    st.divider()

    st.subheader(
        "🧮 Live Awkwardness Score"
    )

    st.markdown(
        f"""
        <div class="score-box">

            <div class="score-number">
                {live_score}/100
            </div>

            <div class="score-label">
                {awkward_label(live_score)}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.progress(
        live_score / 100
    )

    # ========================================================
    # REAL SCORE GRAPH
    # ========================================================

    with engine.lock:

        graph_times = list(
            engine.time_history
        )

        graph_scores = list(
            engine.score_history
        )

    if len(graph_times) > 1:

        import pandas as pd

        graph_data = pd.DataFrame(
            {
                "Time (seconds)": graph_times,
                "Awkwardness": graph_scores
            }
        )

        graph_data = graph_data.set_index(
            "Time (seconds)"
        )

        st.write(
            "### 📈 Awkwardness Over Time"
        )

        st.line_chart(
            graph_data,
            y="Awkwardness",
            height=280
        )

    # ========================================================
    # REAL TIMELINE
    # ========================================================

    st.write(
        "### 🎬 Awkwardness Timeline"
    )

    st.caption(
        "The marker shows the current score "
        "position. The peak records the worst "
        "moment reached during the experiment."
    )

    # --------------------------------------------------------
    # TIMELINE BAR
    # --------------------------------------------------------

    timeline_html = f"""
    <div style="
        position:relative;
        width:100%;
        height:85px;
        border-radius:14px;
        overflow:hidden;
        border:1px solid rgba(255,255,255,0.12);
        display:flex;
        margin-top:10px;
    ">
    """

    sections = [
        ("🙂", "COMFORTABLE", 0, 20),
        ("😐", "SLIGHTLY", 20, 40),
        ("😬", "GETTING", 40, 60),
        ("⚠️", "VERY", 60, 80),
        ("🚨", "ELITE", 80, 90),
        ("👑", "LEGENDARY", 90, 100)
    ]

    for emoji, name, start, end in sections:

        width = end - start

        timeline_html += f"""
        <div style="
            width:{width}%;
            display:flex;
            flex-direction:column;
            align-items:center;
            justify-content:center;
            border-right:1px solid
                rgba(255,255,255,0.10);
            background:
                rgba(255,255,255,0.025);
        ">

            <div style="
                font-size:20px;
            ">
                {emoji}
            </div>

            <div style="
                font-size:10px;
                font-weight:800;
            ">
                {name}
            </div>

            <div style="
                font-size:9px;
                color:#9ca3af;
            ">
                {start}–{end}
            </div>

        </div>
        """

    # --------------------------------------------------------
    # CURRENT SCORE MARKER
    # --------------------------------------------------------

    marker_position = min(
        max(
            live_score,
            0
        ),
        100
    )

    timeline_html += f"""
        <div style="
            position:absolute;
            left:calc({marker_position}% - 2px);
            top:0;
            width:4px;
            height:85px;
            background:white;
            box-shadow:
                0 0 12px rgba(255,255,255,0.8);
            z-index:10;
        ">
        </div>
    """

    timeline_html += "</div>"

    st.markdown(
        timeline_html,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # CURRENT + PEAK
    # --------------------------------------------------------

    with engine.lock:

        peak_score = engine.peak_score
        peak_time = engine.peak_time
        peak_label = engine.peak_label

    t1, t2, t3 = st.columns(3)

    with t1:

        st.metric(
            "📍 Current",
            f"{live_score}/100"
        )

    with t2:

        st.metric(
            "🔥 Peak",
            f"{peak_score}/100"
        )

    with t3:

        st.metric(
            "⏱️ Peak Time",
            f"{peak_time:.1f}s"
        )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    if engine.finished:

        st.divider()

        st.header(
            "📋 FINAL AWKWARDNESS REPORT"
        )

        r1, r2, r3, r4, r5 = st.columns(5)

        with r1:

            st.metric(
                "Final Score",
                f"{engine.final_score}/100"
            )

        with r2:

            st.metric(
                "Longest Silence",
                f"{engine.longest_silence:.2f}s"
            )

        with r3:

            st.metric(
                "Silence Events",
                engine.silence_events
            )

        with r4:

            st.metric(
                "Participants",
                engine.people_count
            )

        with r5:

            st.metric(
                "🔥 Peak",
                f"{engine.peak_score}/100"
            )

        # --------------------------------------------
        # FINAL LABEL
        # --------------------------------------------

        if engine.final_score >= 90:

            st.error(
                f"👑 {engine.final_label}"
            )

        elif engine.final_score >= 80:

            st.warning(
                f"🚨 {engine.final_label}"
            )

        elif engine.final_score >= 60:

            st.warning(
                f"⚠️ {engine.final_label}"
            )

        else:

            st.success(
                f"🙂 {engine.final_label}"
            )

        # --------------------------------------------
        # SCORE BREAKDOWN
        # --------------------------------------------

        final_audio_score = (
            score_silence(
                engine.longest_silence
            )
        )

        final_audio_score += min(
            engine.silence_events * 4,
            20
        )

        final_audio_score = int(
            np.clip(
                final_audio_score,
                0,
                100
            )
        )

        st.write(
            "### 📊 Score Breakdown"
        )

        b1, b2 = st.columns(2)

        with b1:

            st.metric(
                "🎙️ Audio Score",
                f"{final_audio_score}/100"
            )

        with b2:

            st.metric(
                "📷 Vision Score",
                f"{engine.social_score}/100"
            )

        st.info(
            "Final score = 70% audio + 30% vision "
            "when faces are detected. If no face is "
            "detected, audio is used alone."
        )

        # --------------------------------------------
        # PEAK
        # --------------------------------------------

        st.write(
            "### 🏆 Peak Moment"
        )

        st.write(
            f"Peak awkwardness: "
            f"**{engine.peak_score}/100**"
        )

        st.write(
            f"Peak occurred at: "
            f"**{engine.peak_time:.1f} seconds**"
        )

        st.write(
            f"Highest level reached: "
            f"**{engine.peak_label}**"
        )

        # --------------------------------------------
        # INTERPRETATION
        # --------------------------------------------

        st.write(
            "### 🧠 Interpretation"
        )

        if engine.final_score >= 90:

            st.write(
                "The interaction reached an extremely "
                "high awkwardness level according to "
                "the experimental scoring model."
            )

        elif engine.final_score >= 80:

            st.write(
                "The interaction reached the project's "
                "Elite Awkwardness range."
            )

        elif engine.final_score >= 60:

            st.write(
                "The interaction showed a substantial "
                "period of awkward silence."
            )

        elif engine.final_score >= 40:

            st.write(
                "The interaction showed moderate "
                "awkwardness according to the model."
            )

        else:

            st.write(
                "The interaction remained relatively "
                "comfortable."
            )

        st.caption(
            "Facial analysis uses observable cues such "
            "as gaze direction, facing direction and "
            "expression estimates. These cues are not "
            "proof of private emotions, intentions, "
            "attraction or mental state."
        )


# ============================================================
# RUN
# ============================================================

live_dashboard()


# ============================================================
# TECHNICAL ARCHITECTURE
# ============================================================

with st.expander(
    "🔬 Technical Architecture"
):

    st.markdown(
        """
### 🎙️ Audio Pipeline

**Microphone → 44.1 kHz capture → resampling → 16 kHz → Silero VAD → speech/noise/silence**

The original microphone capture remains at
44.1 kHz.

The signal is resampled to 16 kHz for VAD.

Silero processes exact 512-sample chunks.

---

### 📷 Vision Pipeline

**Webcam → OpenCV → MediaPipe Face Landmarker → landmarks + blendshapes → social cues**

The system analyzes multiple participants
and extracts observable visual cues.

---

### 🧮 Multimodal Fusion

When faces are detected:

**70% Audio + 30% Vision**

Without faces:

**100% Audio**

---

### 🏆 Awkwardness Ranking

| Score | Classification |
|---:|---|
| 0–19 | Comfortable |
| 20–39 | Slightly Awkward |
| 40–59 | Getting Awkward |
| 60–79 | Very Awkward |
| 80–89 | 🚨 Elite Awkwardness |
| 90–100 | 👑 Legendary Awkwardness |

---

### 📈 Timeline

The system records the live fused score
throughout the experiment.

It identifies:

- Current awkwardness
- Peak awkwardness
- Time of peak
- Highest ranking reached

---

### ⚠️ Disclaimer

This is an experimental college project.

The vision component measures observable
visual cues and does not scientifically
determine private emotions, intentions,
attraction or mental state.
"""
    )