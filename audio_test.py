import streamlit as st
import sounddevice as sd
import numpy as np
import time
import cv2
import threading


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Silence Compressor™",
    page_icon="🤐",
    layout="wide"
)


# ============================================================
# SETTINGS
# ============================================================

SAMPLE_RATE = 16000
BLOCK_SIZE = 1024
SESSION_TIME = 30

MIN_SILENCE_EVENT = 2.0


# ============================================================
# TITLE
# ============================================================

st.title("🤐 Silence Compressor™")

st.markdown(
    """
    ### The world's most unnecessary conversational AI.

    **Audio + Vision + Context → Awkwardness Score**
    """
)


# ============================================================
# CONTEXT DETECTION
# ============================================================

def detect_context(text):

    text = text.lower().strip()

    if not text:
        return "normal"

    question_words = [
        "why",
        "what",
        "when",
        "where",
        "who",
        "how",
        "did",
        "do you",
        "are you",
        "can you",
        "will you",
        "would you",
        "could you"
    ]

    announcement_words = [
        "attention",
        "please wait",
        "please listen",
        "announcement",
        "the results",
        "next",
        "stand up",
        "remain seated",
        "everyone"
    ]

    if "?" in text:
        return "question"

    for word in question_words:

        if (
            text.startswith(word)
            or f" {word} " in f" {text} "
        ):
            return "question"

    for word in announcement_words:

        if word in text:
            return "announcement"

    return "normal"


# ============================================================
# LIVE AWKWARDNESS
# ============================================================

def calculate_live_awkwardness(silence):

    if silence < 1:
        return 0

    elif silence < 2:
        return 10

    elif silence < 3:
        return 30

    elif silence < 5:
        return 60

    elif silence < 8:
        return 85

    else:
        return 100


# ============================================================
# LIVE COMMENT
# ============================================================

def get_awkward_comment(score):

    if score >= 90:
        return "🚨 KILLING AWKWARD. SAY SOMETHING!"

    elif score >= 70:
        return "💀 BRO... THIS SILENCE IS LOUD."

    elif score >= 50:
        return "😬 Okay... this is getting awkward."

    elif score >= 25:
        return "👀 That's a long pause..."

    else:
        return "😎 Normal conversation."


# ============================================================
# FINAL AUDIO SCORE
# ============================================================

def calculate_audio_score(
    duration,
    previous_silences,
    context
):

    duration_score = min(
        duration * 8,
        60
    )

    repetition_score = min(
        previous_silences * 5,
        20
    )

    extreme_bonus = 0

    if duration >= 8:
        extreme_bonus = 20

    elif duration >= 5:
        extreme_bonus = 10

    if context == "question":
        context_score = 15

    elif context == "announcement":
        context_score = -10

    else:
        context_score = 5

    total = (
        duration_score
        + repetition_score
        + extreme_bonus
        + context_score
    )

    return int(
        max(
            0,
            min(total, 100)
        )
    )


# ============================================================
# CAMERA SCORE
# ============================================================

def calculate_camera_score(
    face_count,
    center_score,
    visibility_score,
    interaction_score
):

    people_score = min(
        face_count * 10,
        20
    )

    engagement_score = int(
        center_score * 0.30
    )

    visibility_component = int(
        visibility_score * 0.20
    )

    interaction_component = int(
        interaction_score * 0.30
    )

    score = (
        people_score
        + engagement_score
        + visibility_component
        + interaction_component
    )

    return int(
        max(
            0,
            min(score, 100)
        )
    )


# ============================================================
# FINAL SCORE
# ============================================================

def calculate_final_score(
    audio_score,
    camera_score
):

    return int(
        audio_score * 0.65
        + camera_score * 0.35
    )


# ============================================================
# CLASSIFICATION
# ============================================================

def classify(score):

    if score < 20:
        return "😌 Comfortable"

    elif score < 40:
        return "🙂 Slightly Awkward"

    elif score < 60:
        return "😐 Getting Awkward"

    elif score < 80:
        return "😬 Very Awkward"

    else:
        return "💀 ELITE AWKWARDNESS"


# ============================================================
# CAMERA CENTER SCORE
# ============================================================

def calculate_center_score(
    x,
    y,
    w,
    h,
    frame_width,
    frame_height
):

    face_center_x = x + w / 2
    face_center_y = y + h / 2

    frame_center_x = frame_width / 2
    frame_center_y = frame_height / 2

    distance_x = abs(
        face_center_x - frame_center_x
    )

    distance_y = abs(
        face_center_y - frame_center_y
    )

    normalized_x = (
        distance_x / frame_width
    )

    normalized_y = (
        distance_y / frame_height
    )

    score = 100 - (
        normalized_x * 100
        + normalized_y * 60
    )

    return max(
        0,
        min(score, 100)
    )


# ============================================================
# VISIBILITY
# ============================================================

def calculate_visibility(
    w,
    h,
    frame_width,
    frame_height
):

    face_area = w * h

    frame_area = (
        frame_width
        * frame_height
    )

    ratio = face_area / frame_area

    score = min(
        ratio * 900,
        100
    )

    return max(
        0,
        min(score, 100)
    )


# ============================================================
# CAMERA ANALYSIS
# ============================================================

def analyze_camera(image):

    frame = cv2.imdecode(
        np.frombuffer(
            image.getvalue(),
            np.uint8
        ),
        cv2.IMREAD_COLOR
    )

    if frame is None:
        return None

    frame = cv2.flip(
        frame,
        1
    )

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    frame_height, frame_width = (
        frame.shape[:2]
    )

    # --------------------------------------------------------
    # HAAR CASCADES
    # --------------------------------------------------------

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades
        + "haarcascade_frontalface_default.xml"
    )

    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades
        + "haarcascade_eye_tree_eyeglasses.xml"
    )

    smile_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades
        + "haarcascade_smile.xml"
    )

    # --------------------------------------------------------
    # FACE DETECTION
    # --------------------------------------------------------

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )

    faces = sorted(
        faces,
        key=lambda f: f[0]
    )

    face_count = len(faces)

    center_scores = []
    visibility_scores = []

    face_info = []

    visual_indicators = []

    visual_tension = 0

    # --------------------------------------------------------
    # NO FACE
    # --------------------------------------------------------

    if face_count == 0:

        cv2.putText(
            frame,
            "NO PARTICIPANTS DETECTED",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )

        return {
            "face_count": 0,
            "center_score": 0,
            "visibility_score": 0,
            "interaction_score": 0,
            "interaction_label": "No participants",
            "visual_tension": 0,
            "visual_indicators": [],
            "camera_score": 0,
            "frame": frame
        }

    # ========================================================
    # EACH FACE
    # ========================================================

    for index, (x, y, w, h) in enumerate(faces):

        center_score = calculate_center_score(
            x,
            y,
            w,
            h,
            frame_width,
            frame_height
        )

        visibility_score = calculate_visibility(
            w,
            h,
            frame_width,
            frame_height
        )

        center_scores.append(
            center_score
        )

        visibility_scores.append(
            visibility_score
        )

        # ----------------------------------------------------
        # FACE BOX
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Person {index + 1}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # FACE ROI
        # ----------------------------------------------------

        face_gray = gray[
            y:y + h,
            x:x + w
        ]

        # ----------------------------------------------------
        # EYES
        # ----------------------------------------------------

        eyes = eye_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(15, 15)
        )

        valid_eyes = []

        for ex, ey, ew, eh in eyes:

            if ey < h * 0.65:

                valid_eyes.append(
                    (ex, ey, ew, eh)
                )

                cv2.rectangle(
                    frame,
                    (
                        x + ex,
                        y + ey
                    ),
                    (
                        x + ex + ew,
                        y + ey + eh
                    ),
                    (255, 255, 0),
                    1
                )

        # ----------------------------------------------------
        # POSSIBLE LOOKING DOWN
        # ----------------------------------------------------

        face_vertical_position = (
            y + h / 2
        ) / frame_height

        if face_vertical_position > 0.68:

            visual_tension += 8

            visual_indicators.append(
                f"Person {index + 1}: "
                "possible looking-down posture 📱"
            )

        # ----------------------------------------------------
        # LOW EYE VISIBILITY
        # ----------------------------------------------------

        if len(valid_eyes) == 0:

            visual_tension += 10

            visual_indicators.append(
                f"Person {index + 1}: "
                "low visible eye activity 👁️"
            )

        # ----------------------------------------------------
        # EYE GEOMETRY
        # ----------------------------------------------------

        if len(valid_eyes) >= 2:

            first_eye = valid_eyes[0]
            second_eye = valid_eyes[1]

            eye1_x = (
                first_eye[0]
                + first_eye[2] / 2
            )

            eye2_x = (
                second_eye[0]
                + second_eye[2] / 2
            )

            eye_distance = abs(
                eye1_x - eye2_x
            )

            if eye_distance < w * 0.18:

                visual_tension += 5

                visual_indicators.append(
                    f"Person {index + 1}: "
                    "unusual eye geometry 👀"
                )

        # ----------------------------------------------------
        # SMILE DETECTION
        # ----------------------------------------------------

        smiles = smile_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.7,
            minNeighbors=20,
            minSize=(25, 15)
        )

        if len(smiles) > 0:

            visual_tension += 10

            visual_indicators.append(
                f"Person {index + 1}: "
                "possible smile / social tension 😬"
            )

        face_info.append(
            {
                "x": x,
                "y": y,
                "w": w,
                "h": h
            }
        )

    # ========================================================
    # INTERACTION
    # ========================================================

    if face_count == 1:

        interaction_score = 35

        interaction_label = (
            "Single participant"
        )

    elif face_count == 2:

        first = face_info[0]
        second = face_info[1]

        first_center = (
            first["x"] + first["w"] / 2,
            first["y"] + first["h"] / 2
        )

        second_center = (
            second["x"] + second["w"] / 2,
            second["y"] + second["h"] / 2
        )

        distance = np.sqrt(
            (
                first_center[0]
                - second_center[0]
            ) ** 2
            +
            (
                first_center[1]
                - second_center[1]
            ) ** 2
        )

        normalized_distance = (
            distance / frame_width
        )

        if normalized_distance < 0.55:

            interaction_score = 85

            interaction_label = (
                "Two-person conversation layout"
            )

        else:

            interaction_score = 60

            interaction_label = (
                "Two participants, separated"
            )

    else:

        interaction_score = 75

        interaction_label = (
            "Group conversation layout"
        )

    # ========================================================
    # GROUP ALIGNMENT
    # ========================================================

    if face_count >= 2:

        centers = []

        for info in face_info:

            centers.append(
                (
                    info["x"] + info["w"] / 2,
                    info["y"] + info["h"] / 2
                )
            )

        group_center_x = np.mean(
            [c[0] for c in centers]
        )

        if abs(
            group_center_x
            - frame_width / 2
        ) < frame_width * 0.25:

            interaction_score += 10

        else:

            interaction_score -= 5

    interaction_score = max(
        0,
        min(interaction_score, 100)
    )

    center_score = (
        sum(center_scores)
        / len(center_scores)
    )

    visibility_score = (
        sum(visibility_scores)
        / len(visibility_scores)
    )

    visual_tension = max(
        0,
        min(visual_tension, 100)
    )

    camera_score = calculate_camera_score(
        face_count,
        center_score,
        visibility_score,
        interaction_score
    )

    # ========================================================
    # HUD
    # ========================================================

    cv2.rectangle(
        frame,
        (10, 10),
        (450, 120),
        (20, 20, 20),
        -1
    )

    cv2.putText(
        frame,
        f"FACES: {face_count}",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"INTERACTION: {interaction_score:.0f}%",
        (25, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"VISUAL TENSION: {visual_tension}",
        (25, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    return {
        "face_count": face_count,
        "center_score": center_score,
        "visibility_score": visibility_score,
        "interaction_score": interaction_score,
        "interaction_label": interaction_label,
        "visual_tension": visual_tension,
        "visual_indicators": visual_indicators,
        "camera_score": camera_score,
        "frame": frame
    }


# ============================================================
# GET MICROPHONE DEVICES
# ============================================================

def get_input_devices():

    devices = sd.query_devices()

    input_devices = []

    for index, device in enumerate(devices):

        if device["max_input_channels"] > 0:

            input_devices.append(
                (
                    index,
                    device["name"]
                )
            )

    return input_devices


# ============================================================
# CONTINUOUS AUDIO ENGINE
# ============================================================

class AudioEngine:

    def __init__(self, device_index):

        self.device_index = device_index

        self.lock = threading.Lock()

        self.current_rms = 0.0

        self.noise_floor = 0.01

        self.threshold = 0.018

        self.stream = None

        self.running = False

    # --------------------------------------------------------
    # CALLBACK
    # --------------------------------------------------------

    def callback(
        self,
        indata,
        frames,
        time_info,
        status
    ):

        if status:

            print(
                "Audio status:",
                status
            )

        if indata is None:

            return

        audio = indata[:, 0]

        if len(audio) == 0:

            return

        rms = float(
            np.sqrt(
                np.mean(
                    audio * audio
                )
            )
        )

        with self.lock:

            self.current_rms = rms

    # --------------------------------------------------------
    # START
    # --------------------------------------------------------

    def start(self):

        self.stream = sd.InputStream(
            device=self.device_index,
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=1,
            dtype="float32",
            callback=self.callback
        )

        self.stream.start()

        self.running = True

    # --------------------------------------------------------
    # READ RMS
    # --------------------------------------------------------

    def get_rms(self):

        with self.lock:

            return self.current_rms

    # --------------------------------------------------------
    # SET THRESHOLD
    # --------------------------------------------------------

    def set_threshold(
        self,
        noise_floor
    ):

        with self.lock:

            self.noise_floor = (
                noise_floor
            )

            self.threshold = max(
                noise_floor * 2.5,
                0.008
            )

    # --------------------------------------------------------
    # GET THRESHOLD
    # --------------------------------------------------------

    def get_threshold(self):

        with self.lock:

            return (
                self.noise_floor,
                self.threshold
            )

    # --------------------------------------------------------
    # STOP
    # --------------------------------------------------------

    def stop(self):

        self.running = False

        if self.stream is not None:

            self.stream.stop()

            self.stream.close()

            self.stream = None


# ============================================================
# CALIBRATE MICROPHONE
# ============================================================

def calibrate_microphone(
    engine,
    seconds=2
):

    levels = []

    start = time.time()

    while (
        time.time() - start
        < seconds
    ):

        levels.append(
            engine.get_rms()
        )

        time.sleep(0.03)

    levels = [
        level
        for level in levels
        if level > 0
    ]

    if not levels:

        noise_floor = 0.01

    else:

        noise_floor = float(
            np.percentile(
                levels,
                40
            )
        )

    engine.set_threshold(
        noise_floor
    )

    return noise_floor


# ============================================================
# AUDIO SESSION
# ============================================================

def run_audio_session(
    device_index
):

    # --------------------------------------------------------
    # UI
    # --------------------------------------------------------

    status_box = st.empty()

    current_box = st.empty()

    score_box = st.empty()

    comment_box = st.empty()

    progress = st.progress(0)

    metrics_box = st.empty()

    timeline_box = st.empty()

    # --------------------------------------------------------
    # ENGINE
    # --------------------------------------------------------

    engine = AudioEngine(
        device_index
    )

    try:

        engine.start()

    except Exception as e:

        st.error(
            "❌ Could not start microphone."
        )

        st.code(
            str(e)
        )

        return None

    try:

        # ----------------------------------------------------
        # CALIBRATION
        # ----------------------------------------------------

        status_box.info(
            "🎤 Calibrating microphone... "
            "Please stay quiet for 2 seconds."
        )

        noise_floor = calibrate_microphone(
            engine,
            2
        )

        noise_floor, threshold = (
            engine.get_threshold()
        )

        status_box.success(
            "🎤 Microphone ready!"
        )

        # ----------------------------------------------------
        # VARIABLES
        # ----------------------------------------------------

        session_start = time.time()

        silence_start = None

        is_speaking = False

        longest_silence = 0.0

        awkward_count = 0

        silence_events = []

        last_event_end = None

        # ----------------------------------------------------
        # MAIN LOOP
        # ----------------------------------------------------

        while True:

            elapsed = (
                time.time()
                - session_start
            )

            if elapsed >= SESSION_TIME:

                break

            # ------------------------------------------------
            # AUDIO LEVEL
            # ------------------------------------------------

            rms = engine.get_rms()

            noise_floor, threshold = (
                engine.get_threshold()
            )

            # ------------------------------------------------
            # SPEECH DETECTION
            # ------------------------------------------------

            speech_detected = (
                rms > threshold
            )

            # =================================================
            # SPEECH
            # =================================================

            if speech_detected:

                status_box.success(
                    "🗣️ SPEAKING"
                )

                # --------------------------------------------
                # SPEECH START
                # --------------------------------------------

                if not is_speaking:

                    is_speaking = True

                    # ----------------------------------------
                    # END SILENCE
                    # ----------------------------------------

                    if silence_start is not None:

                        duration = (
                            time.time()
                            - silence_start
                        )

                        if (
                            duration
                            >= MIN_SILENCE_EVENT
                        ):

                            awkward_count += 1

                            silence_events.append(
                                duration
                            )

                        silence_start = None

                current_silence = 0.0

                current_box.metric(
                    "Current Silence",
                    "0.0 s"
                )

                score_box.metric(
                    "Live Awkwardness",
                    "0 / 100"
                )

                comment_box.success(
                    "😎 Normal conversation."
                )

                # --------------------------------------------
                # SLOWLY LEARN QUIET ENVIRONMENT
                # --------------------------------------------

                if rms < (
                    threshold * 0.8
                ):

                    new_floor = (
                        noise_floor * 0.995
                        + rms * 0.005
                    )

                    engine.set_threshold(
                        new_floor
                    )

            # =================================================
            # SILENCE
            # =================================================

            else:

                status_box.error(
                    "🔇 SILENCE"
                )

                # --------------------------------------------
                # SPEECH → SILENCE
                # --------------------------------------------

                if is_speaking:

                    is_speaking = False

                    silence_start = (
                        time.time()
                    )

                # --------------------------------------------
                # CURRENT SILENCE
                # --------------------------------------------

                if silence_start is not None:

                    current_silence = (
                        time.time()
                        - silence_start
                    )

                    # ----------------------------------------
                    # LIVE SCORE
                    # ----------------------------------------

                    live_score = (
                        calculate_live_awkwardness(
                            current_silence
                        )
                    )

                    comment = (
                        get_awkward_comment(
                            live_score
                        )
                    )

                    # ----------------------------------------
                    # LONGEST
                    # ----------------------------------------

                    if (
                        current_silence
                        > longest_silence
                    ):

                        longest_silence = (
                            current_silence
                        )

                    # ----------------------------------------
                    # UI
                    # ----------------------------------------

                    current_box.metric(
                        "Current Silence",
                        f"{current_silence:.1f} s"
                    )

                    score_box.metric(
                        "Live Awkwardness",
                        f"{live_score} / 100"
                    )

                    if live_score >= 90:

                        comment_box.error(
                            comment
                        )

                    elif live_score >= 50:

                        comment_box.warning(
                            comment
                        )

                    else:

                        comment_box.info(
                            comment
                        )

                    # ----------------------------------------
                    # ADAPTIVE NOISE FLOOR
                    # ----------------------------------------

                    if rms < (
                        threshold * 0.8
                    ):

                        new_floor = (
                            noise_floor * 0.995
                            + rms * 0.005
                        )

                        engine.set_threshold(
                            new_floor
                        )

            # =================================================
            # PROGRESS
            # =================================================

            progress.progress(
                min(
                    elapsed / SESSION_TIME,
                    1.0
                )
            )

            # =================================================
            # METRICS
            # =================================================

            metrics_box.markdown(
                f"""
                **⏱️ Session:** `{elapsed:.1f}s / {SESSION_TIME}s`  
                **🏆 Longest Silence:** `{longest_silence:.1f}s`  
                **⚠️ Awkward Moments:** `{awkward_count}`  
                **📈 RMS Level:** `{rms:.5f}`  
                **📡 Noise Floor:** `{noise_floor:.5f}`  
                **🎚️ Speech Threshold:** `{threshold:.5f}`
                """
            )

            # =================================================
            # TIMELINE
            # =================================================

            if silence_events:

                timeline_text = (
                    "### ⚠️ Awkwardness Timeline\n\n"
                )

                for i, duration in enumerate(
                    silence_events,
                    1
                ):

                    event_score = (
                        calculate_live_awkwardness(
                            duration
                        )
                    )

                    timeline_text += (
                        f"**⚠️ Moment #{i}**  \n"
                        f"Silence: `{duration:.1f}s`  \n"
                        f"Awkwardness: `{event_score}/100`  \n\n"
                    )

                timeline_box.markdown(
                    timeline_text
                )

            # =================================================
            # FAST UI UPDATE
            # =================================================

            time.sleep(0.03)

        # ----------------------------------------------------
        # FINAL SILENCE
        # ----------------------------------------------------

        if silence_start is not None:

            duration = (
                time.time()
                - silence_start
            )

            if duration >= MIN_SILENCE_EVENT:

                awkward_count += 1

                silence_events.append(
                    duration
                )

                longest_silence = max(
                    longest_silence,
                    duration
                )

    finally:

        engine.stop()

    # ========================================================
    # RETURN
    # ========================================================

    return {
        "longest_silence": longest_silence,
        "silence_events": silence_events,
        "awkward_count": awkward_count,
        "noise_floor": noise_floor,
        "threshold": threshold
    }


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "⚙️ Controls"
)

mode = st.sidebar.radio(
    "Select Mode",
    [
        "🎤 Audio",
        "📷 Camera",
        "🧠 Multimodal",
        "🧪 Demo"
    ]
)


# ============================================================
# MICROPHONE SELECTION
# ============================================================

input_devices = get_input_devices()

device_names = [
    name
    for index, name in input_devices
]

if device_names:

    selected_name = st.sidebar.selectbox(
        "🎤 Microphone",
        device_names
    )

    selected_device_index = next(
        index
        for index, name in input_devices
        if name == selected_name
    )

else:

    selected_device_index = None

    st.sidebar.error(
        "No microphone detected."
    )


# ============================================================
# AUDIO MODE
# ============================================================

if mode == "🎤 Audio":

    st.header(
        "🎤 Live Silence Analyzer"
    )

    st.write(
        """
        The microphone continuously measures RMS audio level
        and separates speech from background noise using an
        adaptive threshold.
        """
    )

    context_text = st.text_input(
        "What was said immediately before the silence?",
        placeholder="Example: Why did you do that?"
    )

    context = detect_context(
        context_text
    )

    st.info(
        f"Detected context: **{context.upper()}**"
    )

    if selected_device_index is not None:

        if st.button(
            "▶️ START LIVE AUDIO ANALYSIS"
        ):

            result = run_audio_session(
                selected_device_index
            )

            if result is not None:

                longest = result[
                    "longest_silence"
                ]

                events = result[
                    "silence_events"
                ]

                awkward_count = result[
                    "awkward_count"
                ]

                audio_score = (
                    calculate_audio_score(
                        longest,
                        max(
                            0,
                            len(events) - 1
                        ),
                        context
                    )
                )

                st.divider()

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "Longest Silence",
                    f"{longest:.1f}s"
                )

                c2.metric(
                    "Awkward Moments",
                    awkward_count
                )

                c3.metric(
                    "Audio Score",
                    f"{audio_score}/100"
                )

                c4.metric(
                    "Classification",
                    classify(audio_score)
                )

                if events:

                    st.subheader(
                        "⚠️ Awkwardness Timeline"
                    )

                    for i, duration in enumerate(
                        events,
                        1
                    ):

                        score = (
                            calculate_live_awkwardness(
                                duration
                            )
                        )

                        st.warning(
                            f"⚠️ Moment #{i}  |  "
                            f"Silence: {duration:.1f}s  |  "
                            f"Awkwardness: {score}/100"
                        )


# ============================================================
# CAMERA MODE
# ============================================================

elif mode == "📷 Camera":

    st.header(
        "📷 Visual Interaction Analysis"
    )

    st.info(
        "Take a snapshot of the conversation."
    )

    picture = st.camera_input(
        "Take a conversation snapshot"
    )

    if picture is not None:

        result = analyze_camera(
            picture
        )

        if result is not None:

            st.image(
                cv2.cvtColor(
                    result["frame"],
                    cv2.COLOR_BGR2RGB
                ),
                caption="Analyzed frame",
                use_container_width=True
            )

            st.divider()

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Participants",
                result["face_count"]
            )

            c2.metric(
                "Centre Engagement",
                f"{result['center_score']:.0f}%"
            )

            c3.metric(
                "Visibility",
                f"{result['visibility_score']:.0f}%"
            )

            c4.metric(
                "Interaction",
                f"{result['interaction_score']:.0f}%"
            )

            st.write(
                f"**Layout:** "
                f"{result['interaction_label']}"
            )

            st.metric(
                "Camera Awkwardness",
                f"{result['camera_score']}/100"
            )

            if result[
                "visual_indicators"
            ]:

                st.subheader(
                    "👀 Visual Indicators"
                )

                for indicator in result[
                    "visual_indicators"
                ]:

                    st.write(
                        "• " + indicator
                    )


# ============================================================
# MULTIMODAL MODE
# ============================================================

elif mode == "🧠 Multimodal":

    st.header(
        "🧠 Multimodal Awkwardness Detector"
    )

    st.write(
        """
        **Context + Continuous Audio + Vision**
        """
    )

    previous_statement = st.text_input(
        "What was said before the silence?",
        placeholder="Example: So... are you coming?"
    )

    context = detect_context(
        previous_statement
    )

    st.info(
        f"Detected context: **{context.upper()}**"
    )

    picture = st.camera_input(
        "Take a snapshot of the participants"
    )

    if picture is not None:

        camera_result = analyze_camera(
            picture
        )

        if camera_result is not None:

            st.subheader(
                "📷 Vision Analysis"
            )

            st.image(
                cv2.cvtColor(
                    camera_result["frame"],
                    cv2.COLOR_BGR2RGB
                ),
                caption="Analyzed participants",
                use_container_width=True
            )

            v1, v2, v3, v4 = st.columns(4)

            v1.metric(
                "Participants",
                camera_result["face_count"]
            )

            v2.metric(
                "Interaction",
                f"{camera_result['interaction_score']:.0f}%"
            )

            v3.metric(
                "Visual Tension",
                f"{camera_result['visual_tension']}/100"
            )

            v4.metric(
                "Camera Score",
                f"{camera_result['camera_score']}/100"
            )

            if camera_result[
                "visual_indicators"
            ]:

                for indicator in camera_result[
                    "visual_indicators"
                ]:

                    st.write(
                        "• " + indicator
                    )

            if selected_device_index is not None:

                if st.button(
                    "🚨 START MULTIMODAL ANALYSIS"
                ):

                    st.divider()

                    st.subheader(
                        "🎤 LIVE AUDIO ANALYSIS"
                    )

                    audio_result = (
                        run_audio_session(
                            selected_device_index
                        )
                    )

                    if audio_result is not None:

                        longest = audio_result[
                            "longest_silence"
                        ]

                        events = audio_result[
                            "silence_events"
                        ]

                        awkward_count = (
                            audio_result[
                                "awkward_count"
                            ]
                        )

                        # ------------------------------------
                        # AUDIO
                        # ------------------------------------

                        audio_score = (
                            calculate_audio_score(
                                longest,
                                max(
                                    0,
                                    len(events) - 1
                                ),
                                context
                            )
                        )

                        # ------------------------------------
                        # VISION
                        # ------------------------------------

                        camera_score = (
                            camera_result[
                                "camera_score"
                            ]
                        )

                        # ------------------------------------
                        # FINAL
                        # ------------------------------------

                        final_score = (
                            calculate_final_score(
                                audio_score,
                                camera_score
                            )
                        )

                        classification = (
                            classify(
                                final_score
                            )
                        )

                        # ------------------------------------
                        # PANIC
                        # ------------------------------------

                        panic = (
                            final_score >= 80
                            or longest >= 8
                            or (
                                longest >= 5
                                and camera_result[
                                    "visual_tension"
                                ] >= 40
                            )
                        )

                        # ====================================
                        # RESULT
                        # ====================================

                        st.divider()

                        st.subheader(
                            "📊 FINAL RESULT"
                        )

                        r1, r2, r3, r4 = st.columns(4)

                        r1.metric(
                            "Audio",
                            f"{audio_score}/100"
                        )

                        r2.metric(
                            "Vision",
                            f"{camera_score}/100"
                        )

                        r3.metric(
                            "Longest Silence",
                            f"{longest:.1f}s"
                        )

                        r4.metric(
                            "FINAL",
                            f"{final_score}/100"
                        )

                        st.header(
                            classification
                        )

                        # ====================================
                        # PANIC MODE
                        # ====================================

                        if panic:

                            st.error(
                                "🚨🚨 PANIC MODE ACTIVATED 🚨🚨"
                            )

                            st.markdown(
                                """
                                # ❤️ THUMP... THUMP... THUMP...

                                ## THE SILENCE HAS BECOME TOO POWERFUL.
                                """
                            )

                            st.warning(
                                "Social tension threshold exceeded."
                            )

                        else:

                            st.success(
                                "😌 Situation remains socially survivable."
                            )

                        # ====================================
                        # TIMELINE
                        # ====================================

                        if events:

                            st.divider()

                            st.subheader(
                                "⚠️ AWKWARDNESS TIMELINE"
                            )

                            for i, duration in enumerate(
                                events,
                                1
                            ):

                                event_score = (
                                    calculate_live_awkwardness(
                                        duration
                                    )
                                )

                                st.warning(
                                    f"⚠️ Moment #{i}  |  "
                                    f"Silence: {duration:.1f}s  |  "
                                    f"Awkwardness: "
                                    f"{event_score}/100"
                                )

                        # ====================================
                        # WHY
                        # ====================================

                        st.divider()

                        st.subheader(
                            "🧠 Why was it awkward?"
                        )

                        reasons = []

                        if longest >= 8:

                            reasons.append(
                                f"Extreme silence duration: "
                                f"{longest:.1f}s"
                            )

                        elif longest >= 5:

                            reasons.append(
                                f"Long silence duration: "
                                f"{longest:.1f}s"
                            )

                        elif longest >= 3:

                            reasons.append(
                                f"Noticeable silence: "
                                f"{longest:.1f}s"
                            )

                        if awkward_count >= 3:

                            reasons.append(
                                f"Repeated awkward moments: "
                                f"{awkward_count}"
                            )

                        if context == "question":

                            reasons.append(
                                "Silence followed a question, "
                                "increasing social pressure."
                            )

                        elif context == "announcement":

                            reasons.append(
                                "Announcement context makes "
                                "silence relatively expected."
                            )

                        if camera_result[
                            "face_count"
                        ] >= 2:

                            reasons.append(
                                "Multiple participants were detected."
                            )

                        if camera_result[
                            "interaction_score"
                        ] >= 70:

                            reasons.append(
                                "Participants appeared positioned "
                                "for direct interaction."
                            )

                        if camera_result[
                            "visual_tension"
                        ] >= 30:

                            reasons.append(
                                "Visual tension indicators were detected."
                            )

                        if reasons:

                            for reason in reasons:

                                st.write(
                                    "• " + reason
                                )

                        else:

                            st.write(
                                "No major awkwardness "
                                "factors detected."
                            )

                        # ====================================
                        # SCORE BREAKDOWN
                        # ====================================

                        st.divider()

                        st.subheader(
                            "⚖️ Score Breakdown"
                        )

                        st.write(
                            "🎤 Audio contribution: **65%**"
                        )

                        st.progress(
                            audio_score / 100
                        )

                        st.write(
                            "📷 Vision contribution: **35%**"
                        )

                        st.progress(
                            camera_score / 100
                        )

                        st.caption(
                            "Vision features are heuristic estimates "
                            "from OpenCV face, eye and smile detection. "
                            "They do not determine actual emotions, "
                            "intentions or psychological states."
                        )


# ============================================================
# DEMO MODE
# ============================================================

elif mode == "🧪 Demo":

    st.header(
        "🧪 Demo Mode"
    )

    st.write(
        "Test the scoring system without "
        "microphone or camera."
    )

    demo_duration = st.slider(
        "Silence duration",
        0.0,
        15.0,
        5.0,
        0.5
    )

    demo_events = st.slider(
        "Previous silence events",
        0,
        10,
        2
    )

    demo_context = st.selectbox(
        "Context",
        [
            "normal",
            "question",
            "announcement"
        ]
    )

    demo_camera = st.slider(
        "Camera score",
        0,
        100,
        50
    )

    if st.button(
        "⚡ CALCULATE AWKWARDNESS"
    ):

        audio_score = (
            calculate_audio_score(
                demo_duration,
                demo_events,
                demo_context
            )
        )

        final_score = (
            calculate_final_score(
                audio_score,
                demo_camera
            )
        )

        live_score = (
            calculate_live_awkwardness(
                demo_duration
            )
        )

        st.divider()

        d1, d2, d3, d4 = st.columns(4)

        d1.metric(
            "Audio",
            f"{audio_score}/100"
        )

        d2.metric(
            "Vision",
            f"{demo_camera}/100"
        )

        d3.metric(
            "Live Silence",
            f"{live_score}/100"
        )

        d4.metric(
            "FINAL",
            f"{final_score}/100"
        )

        st.header(
            classify(final_score)
        )

        st.info(
            get_awkward_comment(
                live_score
            )
        )

        if final_score >= 80:

            st.error(
                "🚨 PANIC MODE ACTIVATED"
            )

        elif final_score >= 60:

            st.warning(
                "⚠️ Social tension detected."
            )

        else:

            st.success(
                "😌 Situation is manageable."
            )