import argparse
import time
import logging

from py_pubsub.auto_logger import CameraHubClient

logger = logging.getLogger(__name__)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Record a video for a specified duration and save to an output file.")
    parser.add_argument('--duration', type=float, required=True, help="Duration of the video in seconds.")
    parser.add_argument('--output', type=str, required=True, help="Path to the output video file.")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()

    # Connect to camera hub
    logger.info("Instantiating CameraHubClient")
    camera_hub_client = CameraHubClient(logger)
    logger.info("CameraHubClient created, fetching state")
    state = camera_hub_client.get_state()
    logger.info(f"CameraHubClient connected with state {state}")

    # Start recording
    camera_hub_client.start_recording(args.output, encoder="V4lNvidiaH265")

    # Stop recording after specified duration
    time.sleep(args.duration)
    camera_hub_client.stop_recording()

    print(f"Recorded {args.duration} seconds of video to {args.output}")
