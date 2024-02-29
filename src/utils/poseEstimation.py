import os 
import cv2
from utils import landmarker
import numpy as np
import json
from tqdm import tqdm

PRESET_NAME  = 'preset_01'
TEST_NAME = 'preset_01'
PRESET_VIDEO_PATH = os.path.join('presets',f'{PRESET_NAME}.mp4')
CHUNKS_OUTPUT_PATH = os.path.join('presets',f'{PRESET_NAME}_chunks')
TEST_VIDEO_PATH = os.path.join('presets',f'{TEST_NAME}.mp4')

def start():
    init_chunks_once(PRESET_VIDEO_PATH,CHUNKS_OUTPUT_PATH)
    scores = calculate_score(TEST_VIDEO_PATH,CHUNKS_OUTPUT_PATH)
    print(f"final scores: \n{scores}")


def init_chunks_once( preset_video_path,chunks_output_path,frames_per_chunk = 100):
    print(f'Initializing chunks, frames_per_chunk:{frames_per_chunk}')
    # split_video_into_chunks(chunk_input,keypoints_folder,chunk_duration_second)
    
    # if not exist then create the path, and create chunks
    if not os.path.exists(chunks_output_path):
        # do the split part
        print("Spliting preset into chunks")
        init_chunks(preset_video_path,chunks_output_path,frames_per_chunk)

        #else if existed, we check if the folder is empty or not, if empty we create chunks
    elif len(os.listdir(chunks_output_path)) == 0:
            init_chunks(preset_video_path,chunks_output_path,frames_per_chunk)
    # else just skip
    else:
        print("Skip splitting as chunks folder already has content")


def init_chunks(preset_video_path, chunks_output_path, frames_per_chunk = 100):
    if not os.path.isfile(preset_video_path):
        raise FileNotFoundError('Preset video not found!')
    
    if not os.path.exists(chunks_output_path):
        os.makedirs(chunks_output_path,exist_ok=True)
    
    # read videos
    cap = cv2.VideoCapture(preset_video_path)
    pose_landmarker = landmarker.init_landmarker()
    frame_count = 0

    current_chunks = []
    chunks_count = 0
    with tqdm(desc="Reading video", unit=" frames") as pbar:
        while cap.isOpened():
            ret,frame = cap.read()
            if not ret:
                break
            
            result = landmarker.get_landmark_keypoints(frame, pose_landmarker)
            if result.pose_landmarks:
                keypoints = [[landmark.x, landmark.y, landmark.z] for landmark in result.pose_landmarks[0]]
                current_chunks.append(keypoints)

            frame_count +=1
            ## for debug, check if image correctly have pose detected
            # drawed_frame = landmarker.draw_landmarks_on_image(frame,result)
            # cv2.imshow('Frame_preset',drawed_frame)

            if len(current_chunks) >= frames_per_chunk:
                save_chunks(current_chunks, chunks_output_path, chunks_count)
                chunks_count += 1
                current_chunks = []
            pbar.update(1)

    cap.release()
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def calculate_score(test_video_path,chunks_path,frames_per_chunk = 100):
    if not os.path.isdir(chunks_path):
        raise FileNotFoundError('Chunks folder not found!')
    
    if not os.path.isfile(test_video_path):
        raise FileNotFoundError('Test video not found!')
    

    # read the video frames, perform same function as before, get the video chunks
    cap = cv2.VideoCapture(test_video_path)
    if not cap.isOpened():
        raise RuntimeError('Failed to read test video')
    
    pose_landmarker = landmarker.init_landmarker()
    frame_count = 0
    current_chunks = []
    chunks_count = 0
    all_chunks= []

    with tqdm(desc="Reading test video", unit=" frames") as pbar:
        while cap.isOpened():
            ret,frame = cap.read()
            if not ret:
                break
            result = landmarker.get_landmark_keypoints(frame,pose_landmarker)
            if result.pose_landmarks:
                keypoints = [[landmark.x, landmark.y, landmark.z] for landmark in result.pose_landmarks[0]]
                current_chunks.append(keypoints)

            frame_count +=1
            if len(current_chunks) >= frames_per_chunk:
                    all_chunks.append(current_chunks)
                    chunks_count += 1
                    current_chunks = []
            
            pbar.update(1)
                    
    scores = []
    print(f'chunks_count:{chunks_count}')
    # read the preset chunks list
    with tqdm(desc="Calculating scores", unit=" chunks") as pbar:
        for n in range(chunks_count):
            chunk_preset_name =  f"chunk_{n}.json"
            chunk_preset_path = os.path.join(chunks_path,chunk_preset_name)
            if os.path.isfile(chunk_preset_path):
                preset_keypoints = None 
                with open(chunk_preset_path) as f:
                    preset_keypoints = json.load(f)
                score = calculate_pose_similarity(preset_keypoints,all_chunks[n]) # not sure correct or not
                scores.append(score)

            pbar.update(1)
    
    return scores

# perform computation to check the distance / score if the two video having almost similar pose.
# Thank Gemini for the formula
def calculate_pose_similarity(preset_chunks,test_chunks):
    """
    Calculates pose similarity between two chunks of landmark data.

    Args:
        preset_chunks: List of lists, where each sublist represents the landmark coordinates for a frame in the preset video.
        test_chunks: List of lists, similar to preset_chunks but for the test video.

    Returns:
        A dictionary containing two keys:
            - 'euclidean_distance': The average Euclidean distance between corresponding landmarks in the chunks.
            - 'cosine_similarity': The cosine similarity between the average landmark vectors of the chunks.
    """

    euclidean_distances = []
    cosine_similarities = []

    # print(f'\npreset_chunks:\n{preset_chunks}')
    # print(f'\ntest_chunks:\n{test_chunks}')

    for preset_frame, test_frame in zip(preset_chunks, test_chunks):
        # Convert each frame's landmark data to NumPy arrays for efficient calculations
        preset_array = np.array(preset_frame)
        test_array = np.array(test_frame)

        # Calculate Euclidean distance for each landmark pair
        landmark_distances = np.linalg.norm(preset_array - test_array, axis=1)
        euclidean_distances.append(np.mean(landmark_distances))  # Average individual distances

        # Calculate cosine similarity (use average landmark vectors for robustness)
        preset_avg = np.mean(preset_array, axis=0)
        test_avg = np.mean(test_array, axis=0)
        cosine_similarities.append(np.dot(preset_avg, test_avg) / (np.linalg.norm(preset_avg) * np.linalg.norm(test_avg)))

    # Calculate and return the average scores
    avg_euclidean_distance = np.mean(euclidean_distances)
    avg_cosine_similarity = np.mean(cosine_similarities)

    return {
        "euclidean_distance": avg_euclidean_distance,
        "cosine_similarity": avg_cosine_similarity,
    }
# Utility functions

def save_chunks(current_chunks,chunk_output_path, chunks_count):
    chunk_filename = f"chunk_{chunks_count}.json"
    chunk_filepath = os.path.join(chunk_output_path, chunk_filename)
   
    # np.save(chunk_filepath, current_chunks)
    with open(chunk_filepath, 'w') as f:
        json.dump(current_chunks, f, indent=4)

