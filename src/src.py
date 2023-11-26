import cv2
import base64
import os
import csv
from langchain.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage


def encode_frame(frame):
    """Encode a video frame to base64"""
    _, buffer = cv2.imencode(".jpg", frame)
    return base64.b64encode(buffer).decode("utf-8")

def frame_summarize(frame_base64, prompt):
    """Summarize a video frame"""
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

def generate_frame_summaries(video_path, start_second, end_second, fps=30):
    """
    Generate summaries for specific frames in a video.
    video_path: Path to the video file.
    start_second: Start time in seconds for analysis.
    end_second: End time in seconds for analysis.
    fps: Frame rate of the video.
    """
    video = cv2.VideoCapture(video_path)

    # Calculate start and end frame numbers
    start_frame = start_second * fps
    end_frame = end_second * fps

    prompt = """What’s in this image? Please describe any actions and postures. 
                Format your response in accordance with the guidelines below.
                
                Summary: this should be an overall summary of the image
                Individual 1: this should be a description of the main individual present
                
                If there is more than one individual include them as well, for example:
                Individual 2:, Individual 3:, Individual 4:, etc"""
    frame_count = 0
    output_data = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        if frame_count >= start_frame and frame_count <= end_frame:
            if frame_count % fps == 0:  # Capture one frame per second
                base64_frame = encode_frame(frame)
                summary = frame_summarize(base64_frame, prompt)
                timestamp = frame_count / fps
                output_data.append([timestamp, summary])
        frame_count += 1
        if frame_count > end_frame:
            break

    video.release()
    return output_data

def save_to_csv(data, filename):
    """Save data to a CSV file"""
    with open(filename, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Summary"])
        writer.writerows(data)

# Example usage
video_path = "../data/input/boser.mp4"
output_data = generate_frame_summaries(video_path, 38, 45)
csv_filename = "../data/output/video_summaries.csv"
save_to_csv(output_data, csv_filename)

print(f"Summaries saved to {csv_filename}")
