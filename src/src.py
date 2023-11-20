import cv2
import base64
import requests
import csv


def encode_image(image_path):
    """Encode the image in base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def extract_and_encode_specific_frames(video_path, start_time, end_time, fps):
    """Extract and encode specific frames from a video."""
    encoded_frames = []
    timestamps = []
    start_frame = int(fps * start_time)
    end_frame = int(fps * end_time)
    video = cv2.VideoCapture(video_path)
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    while video.isOpened() and video.get(cv2.CAP_PROP_POS_FRAMES) <= end_frame:
        success, frame = video.read()
        if not success:
            break
        if int(video.get(cv2.CAP_PROP_POS_FRAMES)) % fps == 0:
            timestamp = video.get(cv2.CAP_PROP_POS_FRAMES) / fps
            frame_path = f"../data/frames/frame_{int(timestamp)}.jpg"
            cv2.imwrite(frame_path, frame)
            encoded_frame = encode_image(frame_path)
            encoded_frames.append(encoded_frame)
            timestamps.append(timestamp)

    video.release()
    return encoded_frames, timestamps


def analyze_frames_and_write_to_csv(encoded_frames, timestamps, headers, csv_file):
    """Analyze each frame and write the results to a CSV file."""
    with open(csv_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "analysis"])

        for encoded_frame, timestamp in zip(encoded_frames, timestamps):
            payload = {
                "model": "gpt-4-vision-preview",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """What’s in this image? Please describe any actions and postures. 
                               Format your response in accordance with the guidelines below.
                               
                               Summary: this should be an overall summary of the image

                               Individual 1: this should be a description of the main individual present
                               
                               If there is more than one individual include them as well, for example:
                               
                               Individual 2: """,
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{encoded_frame}"
                                },
                            },
                        ],
                    }
                ],
                "max_tokens": 300,
            }

            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            response_content = (
                response.json()
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            print(response_content)
            writer.writerow([timestamp, response_content])


def main():
    """Main function to execute the script."""
    video_path = "../data/boser.mp4"
    start_time, end_time = 38, 45
    csv_file = "../data/output/analysis_output.csv"

    api_key = ""
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    video.release()

    encoded_frames, timestamps = extract_and_encode_specific_frames(
        video_path, start_time, end_time, fps
    )
    analyze_frames_and_write_to_csv(encoded_frames, timestamps, headers, csv_file)

if __name__ == "__main__":
    main()
