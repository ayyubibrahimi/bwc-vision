import os
from moviepy.editor import VideoFileClip

def extract_frames(video_clip, output_path, fps=30):
    """
    Extract frames from a given video clip at a specified frames per second (fps).

    Args:
    video_clip (VideoFileClip): The video clip from which frames are to be extracted.
    output_path (str): The directory where the extracted frames will be saved.
    fps (int): The number of frames per second to extract. Default is 30.

    The function saves each extracted frame as a JPEG file in the specified output path.
    Each file is named in the format 'frame_{second}_{frame_number}.jpg'.
    """
    duration = video_clip.duration
    for t in range(int(duration)):
        for frame in range(fps):
            time = t + frame / fps
            imgpath = os.path.join(output_path, f"frame_{t}_{frame}.jpg")
            video_clip.save_frame(imgpath, t=time)

def run_extraction(input_path, output_path, fps=30):
    """
    Run the frame extraction process.

    Args:
    input_path (str): Path to the input video file.
    output_path (str): Path to the output directory where frames will be saved.
    fps (int): Frames per second to extract. Default is 30.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Load the video file and extract frames
    clip = VideoFileClip(input_path)
    extract_frames(clip, output_path, fps)
