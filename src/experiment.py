import cv2
import base64
import os
import csv
from datetime import timedelta
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage

def encode_frame(frame):
    """Encode a video frame to base64"""
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")

def analyze_interaction(frame_base64, prompt):
    """Analyze a video frame for interaction"""
    chat = ChatOpenAI(model="gpt-4-vision-preview", max_tokens=1024, api_key="")
    msg = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"},
                    },
                ]
            )
        ]
    )
    return msg.content

def detect_interactions(video_path, start_time, end_time, fps=30):
    """
    Detect interactions in a specific portion of a video.
    video_path: Path to the video file.
    start_time: Start time in seconds for the analysis.
    end_time: End time in seconds for the analysis.
    fps: Frame rate of the video.
    """
    video = cv2.VideoCapture(video_path)
    frame_count = 0
    interaction_intervals = []
    start_interaction = None
    end_interaction = None

    # Calculate start and end frames
    start_frame = start_time * fps
    end_frame = end_time * fps

    # Frame prompt
    prompt = "Determine if the person recording this video is interacting with another individual in this frame."

    while video.isOpened():
        success, frame = video.read()
        if not success or frame_count > end_frame:
            break

        if frame_count >= start_frame:
            if frame_count % fps == 0:  # Analyze one frame per second
                base64_frame = encode_frame(frame)
                interaction = analyze_interaction(base64_frame, prompt)

                if "interacting" in interaction.lower():
                    if start_interaction is None:
                        start_interaction = frame_count / fps
                else:
                    if start_interaction is not None:
                        end_interaction = (frame_count - 1) / fps
                        interaction_intervals.append((start_interaction, end_interaction))
                        start_interaction = None

        frame_count += 1

    video.release()

    # Handle case where interaction is ongoing at the end of the analyzed portion
    if start_interaction is not None:
        interaction_intervals.append((start_interaction, min(end_time, frame_count / fps)))

    return interaction_intervals

def format_interval(interval):
    """Format interval to time string"""
    start, end = interval
    return f"{timedelta(seconds=start)}-{timedelta(seconds=end)}"

def save_to_csv(intervals, filename):
    """Save interaction intervals to a CSV file"""
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Interval"])
        for interval in intervals:
            writer.writerow([format_interval(interval)])

# Example usage
video_path = "../data/input/boser.mp4"
intervals = detect_interactions(video_path, 25, 40)
csv_filename = "../data/output/interaction_intervals.csv"
save_to_csv(intervals, csv_filename)

print(f"Interaction intervals saved to {csv_filename}")
