import cv2
import numpy as np
import os

def extract_key_frames(video_path, output_dir, threshold=30.0):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Read the first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read the first frame.")
        return

    # Convert the first frame to grayscale
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    frame_count = 1
    key_frame_count = 0

    while cap.isOpened():
        # Read the next frame
        ret, curr_frame = cap.read()
        if not ret:
            break

        # Convert the current frame to grayscale
        curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Calculate the absolute difference between the current frame and the previous frame
        frame_diff = cv2.absdiff(curr_frame_gray, prev_frame_gray)

        # Calculate the mean of the frame difference
        mean_diff = np.mean(frame_diff)

        # If the mean difference is above the threshold, save the frame as a key frame
        if mean_diff > threshold:
            key_frame_path = os.path.join(output_dir, f"key_frame_{key_frame_count + 1:04d}.jpg")
            cv2.imwrite(key_frame_path, curr_frame)
            key_frame_count += 1

        # Update the previous frame
        prev_frame_gray = curr_frame_gray

        frame_count += 1

    cap.release()
    print(f"Extracted {key_frame_count} key frames from the video.")

# Example usage
extract_key_frames("./videos/Mark_AI_Security.mp4", "output_key_frames")
