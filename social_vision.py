import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "face_landmarker.task"

CAMERA_INDEX = 0

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 30

MAX_PEOPLE = 4


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions

FaceLandmarker = mp.tasks.vision.FaceLandmarker

FaceLandmarkerOptions = (
    mp.tasks.vision.FaceLandmarkerOptions
)

RunningMode = mp.tasks.vision.RunningMode


options = FaceLandmarkerOptions(

    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),

    running_mode=RunningMode.VIDEO,

    num_faces=MAX_PEOPLE,

    min_face_detection_confidence=0.5,

    min_face_presence_confidence=0.5,

    min_tracking_confidence=0.5,

    output_face_blendshapes=True,

    output_facial_transformation_matrixes=True
)


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_INDEX
)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    FRAME_WIDTH
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    FRAME_HEIGHT
)

cap.set(
    cv2.CAP_PROP_FPS,
    TARGET_FPS
)


if not cap.isOpened():

    print("❌ Camera could not be opened.")

    raise SystemExit


# ============================================================
# PERSON STATE
# ============================================================

class PersonState:

    def __init__(self):

        self.frames = 0

        self.eye_contact_frames = 0

        self.look_away_frames = 0

        self.look_down_frames = 0

        self.smile_frames = 0

        self.blink_frames = 0

        self.gaze_changes = 0

        self.interaction_frames = 0

        self.mutual_interaction_frames = 0

        self.last_gaze = "CENTER"

        self.last_interaction = False

        self.gaze_history = deque(
            maxlen=120
        )

        self.smile_history = deque(
            maxlen=120
        )

        self.redness_history = deque(
            maxlen=120
        )

        self.score_history = deque(
            maxlen=120
        )


people = [
    PersonState()
    for _ in range(MAX_PEOPLE)
]


# ============================================================
# BLENDSHAPE HELPER
# ============================================================

def get_shape(
    shapes,
    name
):

    for item in shapes:

        if item.category_name == name:

            return float(
                item.score
            )

    return 0.0


# ============================================================
# EXPRESSION
# ============================================================

def estimate_expression(
    smile,
    frown,
    brow_up,
    brow_down,
    mouth_open
):

    if smile > 0.35:

        return "HAPPY"

    if frown > 0.30:

        return "SAD"

    if (
        brow_up > 0.40
        and mouth_open > 0.35
    ):

        return "SURPRISED"

    if (
        brow_down > 0.35
        and frown > 0.20
    ):

        return "ANGRY"

    return "NEUTRAL"


# ============================================================
# FACIAL REDNESS
# ============================================================

def calculate_redness(
    frame,
    x1,
    y1,
    x2,
    y2
):

    h, w = frame.shape[:2]

    x1 = max(
        0,
        x1
    )

    y1 = max(
        0,
        y1
    )

    x2 = min(
        w - 1,
        x2
    )

    y2 = min(
        h - 1,
        y2
    )


    if x2 <= x1 or y2 <= y1:

        return 0.0


    face = frame[
        y1:y2,
        x1:x2
    ]


    if face.size == 0:

        return 0.0


    fh, fw = face.shape[:2]


    # Approximate cheek region.

    left = int(
        fw * 0.20
    )

    right = int(
        fw * 0.80
    )

    top = int(
        fh * 0.48
    )

    bottom = int(
        fh * 0.78
    )


    cheek = face[
        top:bottom,
        left:right
    ]


    if cheek.size == 0:

        return 0.0


    rgb = cv2.cvtColor(
        cheek,
        cv2.COLOR_BGR2RGB
    ).astype(
        np.float32
    )


    r = rgb[:, :, 0]

    g = rgb[:, :, 1]

    b = rgb[:, :, 2]


    redness = r - (
        (g + b) / 2
    )


    return float(
        np.mean(redness)
    )


# ============================================================
# HEAD / FACE DIRECTION
# ============================================================

def estimate_face_direction(
    landmarks,
    width,
    height
):

    """
    Estimate horizontal face direction using the nose
    position relative to the eye-line.

    This is intentionally a behavioural approximation,
    not a precise 3D head-pose measurement.
    """

    # MediaPipe landmark indices:
    #
    # 1   = nose region
    # 33  = left eye region
    # 263 = right eye region

    nose = landmarks[1]

    left_eye = landmarks[33]

    right_eye = landmarks[263]


    nose_x = nose.x * width

    eye_center_x = (
        left_eye.x * width
        +
        right_eye.x * width
    ) / 2


    eye_distance = abs(
        (
            right_eye.x
            -
            left_eye.x
        ) * width
    )


    if eye_distance < 1:

        return "CENTER", 0.0


    # Normalized horizontal displacement.

    yaw_signal = (
        nose_x - eye_center_x
    ) / eye_distance


    # Thresholds deliberately include a dead-zone
    # to prevent jitter.

    if yaw_signal > 0.18:

        return "RIGHT", yaw_signal


    if yaw_signal < -0.18:

        return "LEFT", yaw_signal


    return "CENTER", yaw_signal


# ============================================================
# SOCIAL SCORE
# ============================================================

def calculate_social_score(
    person
):

    if person.frames < 20:

        return 0


    eye_avoidance = (
        person.look_away_frames
        /
        person.frames
    )


    looking_down = (
        person.look_down_frames
        /
        person.frames
    )


    gaze_switch_score = min(
        person.gaze_changes / 15,
        1.0
    )


    interaction_score = (
        person.interaction_frames
        /
        person.frames
    )


    smile_ratio = (
        person.smile_frames
        /
        person.frames
    )


    # --------------------------------------------------------
    # Weighted behavioural score
    # --------------------------------------------------------

    score = (

        eye_avoidance
        * 30

        +

        looking_down
        * 20

        +

        gaze_switch_score
        * 20

        +

        interaction_score
        * 15

        +

        smile_ratio
        * 10

        +

        min(
            person.blink_frames
            /
            person.frames,
            0.5
        )
        * 10
    )


    return int(
        min(
            score,
            100
        )
    )


# ============================================================
# DETERMINE WHO SOMEONE IS FACING
# ============================================================

def determine_target(
    current_index,
    direction,
    centers
):

    if len(centers) <= 1:

        return None


    current_x = centers[
        current_index
    ][0]


    best_target = None

    best_distance = float(
        "inf"
    )


    for i, center in enumerate(
        centers
    ):

        if i == current_index:

            continue


        target_x = center[0]

        difference = (
            target_x
            -
            current_x
        )


        # Person is to the right.

        if (
            direction == "RIGHT"
            and difference > 0
        ):

            distance = abs(
                difference
            )

            if distance < best_distance:

                best_distance = distance

                best_target = i


        # Person is to the left.

        elif (
            direction == "LEFT"
            and difference < 0
        ):

            distance = abs(
                difference
            )

            if distance < best_distance:

                best_distance = distance

                best_target = i


    return best_target


# ============================================================
# START MEDIAPIPE
# ============================================================

with FaceLandmarker.create_from_options(
    options
) as landmarker:


    previous_time = time.time()


    while True:

        success, frame = cap.read()


        if not success:

            print(
                "❌ Camera frame unavailable."
            )

            break


        # ----------------------------------------------------
        # Mirror
        # ----------------------------------------------------

        frame = cv2.flip(
            frame,
            1
        )


        height, width = frame.shape[:2]


        # ----------------------------------------------------
        # Convert image
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )


        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        timestamp = int(
            time.time() * 1000
        )


        # ----------------------------------------------------
        # Detect
        # ----------------------------------------------------

        result = (
            landmarker.detect_for_video(
                mp_image,
                timestamp
            )
        )


        face_count = len(
            result.face_landmarks
        )


        # ====================================================
        # FIRST PASS
        # Get face boxes + centers
        # ====================================================

        face_data = []

        centers = []


        for i, landmarks in enumerate(
            result.face_landmarks
        ):

            xs = np.array([
                p.x * width
                for p in landmarks
            ])

            ys = np.array([
                p.y * height
                for p in landmarks
            ])


            x1 = int(
                max(
                    0,
                    np.min(xs)
                )
            )

            y1 = int(
                max(
                    0,
                    np.min(ys)
                )
            )

            x2 = int(
                min(
                    width - 1,
                    np.max(xs)
                )
            )

            y2 = int(
                min(
                    height - 1,
                    np.max(ys)
                )
            )


            center_x = int(
                (x1 + x2) / 2
            )

            center_y = int(
                (y1 + y2) / 2
            )


            centers.append(
                (
                    center_x,
                    center_y
                )
            )


            face_data.append({
                "landmarks": landmarks,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            })


        # ====================================================
        # SECOND PASS
        # Analyse each person
        # ====================================================

        person_results = []


        for i, data in enumerate(
            face_data
        ):

            if i >= MAX_PEOPLE:

                break


            person = people[i]

            person.frames += 1


            landmarks = data[
                "landmarks"
            ]


            x1 = data["x1"]

            y1 = data["y1"]

            x2 = data["x2"]

            y2 = data["y2"]


            # ------------------------------------------------
            # Blendshapes
            # ------------------------------------------------

            shapes = []

            if (
                result.face_blendshapes
                and i
                <
                len(
                    result.face_blendshapes
                )
            ):

                shapes = (
                    result.face_blendshapes[i]
                )


            smile = (

                get_shape(
                    shapes,
                    "mouthSmileLeft"
                )

                +

                get_shape(
                    shapes,
                    "mouthSmileRight"
                )

            ) / 2


            frown = (

                get_shape(
                    shapes,
                    "mouthFrownLeft"
                )

                +

                get_shape(
                    shapes,
                    "mouthFrownRight"
                )

            ) / 2


            blink = (

                get_shape(
                    shapes,
                    "eyeBlinkLeft"
                )

                +

                get_shape(
                    shapes,
                    "eyeBlinkRight"
                )

            ) / 2


            look_down = (

                get_shape(
                    shapes,
                    "eyeLookDownLeft"
                )

                +

                get_shape(
                    shapes,
                    "eyeLookDownRight"
                )

            ) / 2


            look_left = (

                get_shape(
                    shapes,
                    "eyeLookInLeft"
                )

                +

                get_shape(
                    shapes,
                    "eyeLookOutRight"
                )

            ) / 2


            look_right = (

                get_shape(
                    shapes,
                    "eyeLookOutLeft"
                )

                +

                get_shape(
                    shapes,
                    "eyeLookInRight"
                )

            ) / 2


            brow_up = (

                get_shape(
                    shapes,
                    "browInnerUp"
                )

                +

                get_shape(
                    shapes,
                    "browOuterUpLeft"
                )

                +

                get_shape(
                    shapes,
                    "browOuterUpRight"
                )

            ) / 3


            brow_down = (

                get_shape(
                    shapes,
                    "browDownLeft"
                )

                +

                get_shape(
                    shapes,
                    "browDownRight"
                )

            ) / 2


            mouth_open = get_shape(
                shapes,
                "jawOpen"
            )


            # ------------------------------------------------
            # Gaze classification
            # ------------------------------------------------

            if look_down > 0.25:

                gaze = "DOWN"

                person.look_down_frames += 1


            elif look_left > 0.25:

                gaze = "LEFT"

                person.look_away_frames += 1


            elif look_right > 0.25:

                gaze = "RIGHT"

                person.look_away_frames += 1


            else:

                gaze = "CENTER"

                person.eye_contact_frames += 1


            # ------------------------------------------------
            # Gaze transitions
            # ------------------------------------------------

            if (
                gaze
                !=
                person.last_gaze
            ):

                if (
                    gaze != "CENTER"
                    and
                    person.last_gaze
                    == "CENTER"
                ):

                    person.gaze_changes += 1


                person.last_gaze = gaze


            person.gaze_history.append(
                gaze
            )


            # ------------------------------------------------
            # Smile
            # ------------------------------------------------

            if smile > 0.25:

                person.smile_frames += 1


            person.smile_history.append(
                smile
            )


            # ------------------------------------------------
            # Blink
            # ------------------------------------------------

            if blink > 0.55:

                person.blink_frames += 1


            # ------------------------------------------------
            # Facial redness
            # ------------------------------------------------

            redness = calculate_redness(
                frame,
                x1,
                y1,
                x2,
                y2
            )


            person.redness_history.append(
                redness
            )


            if len(
                person.redness_history
            ) >= 30:

                baseline = np.median(
                    list(
                        person.redness_history
                    )[:20]
                )

                redness_change = (
                    redness
                    -
                    baseline
                )

            else:

                redness_change = 0.0


            # ------------------------------------------------
            # Expression
            # ------------------------------------------------

            expression = (
                estimate_expression(
                    smile,
                    frown,
                    brow_up,
                    brow_down,
                    mouth_open
                )
            )


            # ------------------------------------------------
            # Face direction
            # ------------------------------------------------

            face_direction, yaw = (
                estimate_face_direction(
                    landmarks,
                    width,
                    height
                )
            )


            # ------------------------------------------------
            # Find potential person being faced
            # ------------------------------------------------

            target = determine_target(
                i,
                face_direction,
                centers
            )


            interaction = (
                target is not None
                and
                face_direction != "CENTER"
            )


            if interaction:

                person.interaction_frames += 1


            person_results.append({

                "index": i,

                "gaze": gaze,

                "direction":
                    face_direction,

                "target":
                    target,

                "interaction":
                    interaction,

                "smile":
                    smile,

                "blink":
                    blink,

                "look_down":
                    look_down,

                "expression":
                    expression,

                "redness":
                    redness_change,

                "yaw":
                    yaw,

                "x1": x1,

                "y1": y1,

                "x2": x2,

                "y2": y2

            })


        # ====================================================
        # MUTUAL INTERACTION
        # ====================================================

        for current in person_results:

            current_index = (
                current["index"]
            )

            target = current[
                "target"
            ]


            if target is None:

                continue


            for other in person_results:

                if (
                    other["index"]
                    ==
                    target
                ):

                    if (
                        other["target"]
                        ==
                        current_index
                    ):

                        people[
                            current_index
                        ].mutual_interaction_frames += 1


                        people[
                            target
                        ].mutual_interaction_frames += 1


        # ====================================================
        # DRAW PEOPLE
        # ====================================================

        for data in person_results:

            i = data["index"]


            x1 = data["x1"]

            y1 = data["y1"]

            x2 = data["x2"]

            y2 = data["y2"]


            # ------------------------------------------------
            # Face box
            # ------------------------------------------------

            cv2.rectangle(

                frame,

                (x1, y1),

                (x2, y2),

                (0, 255, 0),

                2

            )


            # ------------------------------------------------
            # Social score
            # ------------------------------------------------

            score = calculate_social_score(
                people[i]
            )


            people[i].score_history.append(
                score
            )


            # ------------------------------------------------
            # Text position
            # ------------------------------------------------

            tx = x1

            ty = max(
                25,
                y1 - 135
            )


            # ------------------------------------------------
            # Person label
            # ------------------------------------------------

            cv2.putText(

                frame,

                f"PERSON {i + 1}",

                (tx, ty),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.60,

                (0, 255, 0),

                2

            )


            # ------------------------------------------------
            # Gaze
            # ------------------------------------------------

            cv2.putText(

                frame,

                f"Gaze: {data['gaze']}",

                (tx, ty + 21),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.48,

                (255, 255, 255),

                2

            )


            # ------------------------------------------------
            # Direction
            # ------------------------------------------------

            cv2.putText(

                frame,

                f"Facing: {data['direction']}",

                (tx, ty + 41),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.48,

                (255, 255, 255),

                2

            )


            # ------------------------------------------------
            # Expression
            # ------------------------------------------------

            cv2.putText(

                frame,

                f"Expression: {data['expression']}",

                (tx, ty + 61),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.48,

                (255, 255, 255),

                2

            )


            # ------------------------------------------------
            # Smile
            # ------------------------------------------------

            cv2.putText(

                frame,

                f"Smile: {data['smile']:.2f}",

                (tx, ty + 81),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.48,

                (255, 255, 255),

                2

            )


            # ------------------------------------------------
            # Facial colour
            # ------------------------------------------------

            cv2.putText(

                frame,

                f"Face warmth: {data['redness']:+.1f}",

                (tx, ty + 101),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.48,

                (255, 255, 255),

                2

            )


            # ------------------------------------------------
            # Social score
            # ------------------------------------------------

            cv2.putText(

                frame,

                f"Social cue: {score}/100",

                (tx, ty + 121),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.48,

                (0, 255, 255),

                2

            )


            # ------------------------------------------------
            # Interaction line
            # ------------------------------------------------

            target = data["target"]


            if target is not None:

                target_center = centers[
                    target
                ]

                person_center = centers[
                    i
                ]


                cv2.arrowedLine(

                    frame,

                    person_center,

                    target_center,

                    (255, 0, 255),

                    2,

                    tipLength=0.15

                )


        # ====================================================
        # MUTUAL INTERACTION HUD
        # ====================================================

        mutual_pairs = []


        for data in person_results:

            target = data[
                "target"
            ]


            if target is None:

                continue


            for other in person_results:

                if (
                    other["index"]
                    == target
                    and
                    other["target"]
                    == data["index"]
                ):

                    pair = tuple(
                        sorted(
                            [
                                data["index"],
                                target
                            ]
                        )
                    )


                    if pair not in mutual_pairs:

                        mutual_pairs.append(
                            pair
                        )


        # ====================================================
        # FPS
        # ====================================================

        now = time.time()


        fps = 1 / max(
            now - previous_time,
            0.001
        )


        previous_time = now


        # ====================================================
        # HUD
        # ====================================================

        cv2.rectangle(

            frame,

            (0, 0),

            (520, 125),

            (0, 0, 0),

            -1

        )


        cv2.putText(

            frame,

            f"PEOPLE: {face_count}",

            (15, 28),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.70,

            (255, 255, 255),

            2

        )


        cv2.putText(

            frame,

            f"FPS: {fps:.1f}",

            (15, 55),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.60,

            (255, 255, 255),

            2

        )


        cv2.putText(

            frame,

            "SOCIAL CUE ENGINE: ACTIVE",

            (15, 82),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.55,

            (0, 255, 0),

            2

        )


        if mutual_pairs:

            interaction_text = (
                "MUTUAL INTERACTION: YES"
            )

        else:

            interaction_text = (
                "MUTUAL INTERACTION: NO"
            )


        cv2.putText(

            frame,

            interaction_text,

            (15, 108),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.52,

            (255, 0, 255),

            2

        )


        # ====================================================
        # SHOW
        # ====================================================

        cv2.imshow(

            "Silence Compressor - Social Intelligence",

            frame

        )


        # ====================================================
        # EXIT
        # ====================================================

        key = cv2.waitKey(1) & 0xFF


        if key == 27:

            break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()


# ============================================================
# FINAL SESSION REPORT
# ============================================================

print()
print(
    "=============================================="
)

print(
    "      SILENCE COMPRESSOR - VISION REPORT"
)

print(
    "=============================================="
)


for i, person in enumerate(
    people
):

    if person.frames < 20:

        continue


    score = calculate_social_score(
        person
    )


    eye_contact = (

        person.eye_contact_frames
        /
        person.frames

    ) * 100


    looking_down = (

        person.look_down_frames
        /
        person.frames

    ) * 100


    smile_activity = (

        person.smile_frames
        /
        person.frames

    ) * 100


    interaction = (

        person.interaction_frames
        /
        person.frames

    ) * 100


    mutual = (

        person.mutual_interaction_frames
        /
        person.frames

    ) * 100


    print()

    print(
        f"PERSON {i + 1}"
    )

    print(
        f"Eye-contact tendency : "
        f"{eye_contact:.1f}%"
    )

    print(
        f"Looking-down         : "
        f"{looking_down:.1f}%"
    )

    print(
        f"Smile activity       : "
        f"{smile_activity:.1f}%"
    )

    print(
        f"Gaze changes         : "
        f"{person.gaze_changes}"
    )

    print(
        f"Person interaction   : "
        f"{interaction:.1f}%"
    )

    print(
        f"Mutual interaction   : "
        f"{mutual:.1f}%"
    )

    print(
        f"Social cue score     : "
        f"{score}/100"
    )


print()

print(
    "NOTE:"
)

print(
    "These measurements represent observable "
    "facial/behavioural cues."
)

print(
    "They do not establish a person's actual "
    "emotion, attraction, or intention."
)

print(
    "=============================================="
)