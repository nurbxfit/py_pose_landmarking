import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import numpy as np
import os 

MODEL_PATH = os.path.join('models','pose_landmarker_full.task')

def init_landmarker(model_path=MODEL_PATH,running_mode = 'image'):
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult
    VisionRunningMode = mp.tasks.vision.RunningMode

    RunningMode = VisionRunningMode.VIDEO if running_mode == 'video' else VisionRunningMode.IMAGE
    # Create a pose landmarker instance with the video mode:
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode)
    
    return PoseLandmarker.create_from_options(options)


# bite frane is numpy array
def get_landmark_keypoints(frame,landmarker, timestamp = 0):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # pose_landmarker_result = landmarker.detect_for_video(mp_image, timestamp)
    result = landmarker.detect(mp_image)
    return result 


def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image