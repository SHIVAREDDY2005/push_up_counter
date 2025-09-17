# import cv2
# import mediapipe as mp
# import tempfile
# import streamlit as st

# # Mediapipe setup
# mp_draw = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mp_pose = mp.solutions.pose

# # ---------------- Processing Functions ----------------
# def process_video(video_path, placeholder):
#     count = 0
#     position = None
#     cap = cv2.VideoCapture(video_path)

#     with mp_pose.Pose(min_detection_confidence=0.7,
#                       min_tracking_confidence=0.7) as pose:
#         # while cap.isOpened():
#             # success, image = cap.read()
#             # if not success:
#             #     break

#             rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             results = pose.process(rgb_image)

#             if results.pose_landmarks:
#                 mp_draw.draw_landmarks(
#                     image,
#                     results.pose_landmarks,
#                     mp_pose.POSE_CONNECTIONS,
#                     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
#                 )

#                 # Extract landmarks
#                 h, w, _ = image.shape
#                 imlist = []
#                 for id, lm in enumerate(results.pose_landmarks.landmark):
#                     imlist.append([id, int(lm.x * w), int(lm.y * h)])

#                 # Push-up logic
#                 if len(imlist) >= 15:
#                     if imlist[12][2] >= imlist[14][2] and imlist[11][2] >= imlist[13][2]:
#                         position = "down"
#                     if position == "down" and imlist[12][2] < imlist[14][2] and imlist[11][2] < imlist[13][2]:
#                         position = "up"
#                         count += 1

#             cv2.putText(image, f'Push-ups: {count}', (30, 60),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

#             placeholder.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB")

#         cap.release()
#     return count


# def live_counter(placeholder, stop_flag):
#     count = 0
#     position = None
#     cap = cv2.VideoCapture(0)  # webcam

#     with mp_pose.Pose(min_detection_confidence=0.7,
#                       min_tracking_confidence=0.7) as pose:
#         while cap.isOpened():
#             if stop_flag():   # 🔴 check if stop button pressed
#                 break

#             success, image = cap.read()
#             if not success:
#                 break

#             rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#             results = pose.process(rgb_image)

#             if results.pose_landmarks:
#                 mp_draw.draw_landmarks(
#                     image,
#                     results.pose_landmarks,
#                     mp_pose.POSE_CONNECTIONS,
#                     landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
#                 )

#                 h, w, _ = image.shape
#                 imlist = []
#                 for id, lm in enumerate(results.pose_landmarks.landmark):
#                     imlist.append([id, int(lm.x * w), int(lm.y * h)])

#                 if len(imlist) >= 15:
#                     if imlist[12][2] >= imlist[14][2] and imlist[11][2] >= imlist[13][2]:
#                         position = "down"
#                     if position == "down" and imlist[12][2] < imlist[14][2] and imlist[11][2] < imlist[13][2]:
#                         position = "up"
#                         count += 1

#             cv2.putText(image, f'Push-ups: {count}', (30, 60),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

#             placeholder.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), channels="RGB")

#     cap.release()
#     return count


# # ---------------- Streamlit UI ----------------
# st.title("🏋️ Push-up Counter with Mediapipe")

# mode = st.radio("Choose mode:", ["📹 Upload Video", "🎥 Live Camera"])

# if mode == "📹 Upload Video":
#     uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
#     if uploaded_file is not None:
#         tfile = tempfile.NamedTemporaryFile(delete=False)
#         tfile.write(uploaded_file.read())
#         st.info("Processing video... ⏳")
#         placeholder = st.empty()
#         total_count = process_video(tfile.name, placeholder)
#         st.success(f"✅ Done! Total push-ups counted: {total_count}")

# elif mode == "🎥 Live Camera":
#     st.info("Live mode started... do push-ups in front of your webcam 🎥")

#     placeholder = st.empty()

#     # stop button logic
#     stop_pressed = st.button("🛑 Stop Live Session")

#     def stop_flag():
#         return st.session_state.get("stop", False)

#     if stop_pressed:
#         st.session_state["stop"] = True
#     else:
#         st.session_state["stop"] = False

#     total_count = live_counter(placeholder, stop_flag)
#     st.success(f"✅ Session ended! Total push-ups counted: {total_count}")







import cv2
import mediapipe as mp
import tempfile
import streamlit as st
import numpy as np
from PIL import Image

# ---------------- Mediapipe Setup ----------------
mp_draw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# ---------------- Processing Functions ----------------
def process_frame(image, count_data):
    """
    Process a single frame for push-up counting
    count_data: dict with 'count' and 'position'
    """
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_pose.Pose(min_detection_confidence=0.7,
                      min_tracking_confidence=0.7) as pose:
        results = pose.process(rgb_image)

        if results.pose_landmarks:
            mp_draw.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

            h, w, _ = image.shape
            imlist = []
            for id, lm in enumerate(results.pose_landmarks.landmark):
                imlist.append([id, int(lm.x * w), int(lm.y * h)])

            if len(imlist) >= 15:
                # Push-up logic
                if imlist[12][2] >= imlist[14][2] and imlist[11][2] >= imlist[13][2]:
                    count_data['position'] = "down"
                if count_data['position'] == "down" and imlist[12][2] < imlist[14][2] and imlist[11][2] < imlist[13][2]:
                    count_data['position'] = "up"
                    count_data['count'] += 1

    cv2.putText(image, f'Push-ups: {count_data["count"]}', (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    return image

def process_video(video_path, placeholder):
    cap = cv2.VideoCapture(video_path)
    count_data = {'count': 0, 'position': None}
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame = process_frame(frame, count_data)
        placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
    cap.release()
    return count_data['count']

def live_counter(placeholder):
    """
    Live webcam using st.camera_input
    """
    st.info("Take repeated snapshots from your webcam. Count will update automatically.")
    count_data = {'count': 0, 'position': None}
    
    while True:
        img_file_buffer = st.camera_input("Capture frame")
        if img_file_buffer is None:
            break  # stop if user cancels
        # Convert to OpenCV image
        img = Image.open(img_file_buffer)
        frame = np.array(img)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        frame = process_frame(frame, count_data)
        placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
    
    return count_data['count']

# ---------------- Streamlit UI ----------------
st.title("🏋️ Push-up Counter with Mediapipe")

mode = st.radio("Choose mode:", ["📹 Upload Video", "🎥 Live Camera"])

if mode == "📹 Upload Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        st.info("Processing video... ⏳")
        placeholder = st.empty()
        total_count = process_video(tfile.name, placeholder)
        st.success(f"✅ Done! Total push-ups counted: {total_count}")

elif mode == "🎥 Live Camera":
    st.info("Live webcam mode using snapshots 🎥")
    placeholder = st.empty()
    total_count = live_counter(placeholder)
    st.success(f"✅ Session ended! Total push-ups counted: {total_count}")
