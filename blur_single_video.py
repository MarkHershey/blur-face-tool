from pathlib import Path

import cv2
import face_recognition


def check_single_video(video_filepath: str, batch_size: int = 3) -> bool:
    assert Path(video_filepath).is_file()
    # Open video file
    video_capture = cv2.VideoCapture(video_filepath)

    frames = []
    frame_count = 0
    BATCH_SIZE = batch_size

    while video_capture.isOpened():
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Bail out when the video file ends
        if not ret:
            break

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        frame = frame[:, :, ::-1]

        # Save each frame of the video to a list
        frame_count += 1
        frames.append(frame)

        # Every BATCH_SIZE frames (the default batch size), batch process the list of frames to find faces
        if len(frames) == BATCH_SIZE:
            batch_of_face_locations = face_recognition.batch_face_locations(
                frames, number_of_times_to_upsample=0
            )

            # Now let's list all the faces we found in all BATCH_SIZE frames
            for frame_number_in_batch, face_locations in enumerate(
                batch_of_face_locations
            ):
                number_of_faces_in_frame = len(face_locations)

                if number_of_faces_in_frame >= 1:
                    return True

                frame_number = frame_count - BATCH_SIZE + frame_number_in_batch
                print(
                    f"Found {number_of_faces_in_frame} face(s) in frame #{frame_number}."
                )

                for face_location in face_locations:
                    # Print the location of each face in this frame
                    top, right, bottom, left = face_location
                    print(
                        " - A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(
                            top, left, bottom, right
                        )
                    )

            # Clear the frames array to start the next batch
            frames = []

    return False


if __name__ == "__main__":
    pass
