import face_recognition
import cv2


# Open video file
video_capture = cv2.VideoCapture("sample_videos/IMG_0095.MOV")

frames = []
frame_count = 0

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

    # Every 128 frames (the default batch size), batch process the list of frames to find faces
    if len(frames) == 128:
        batch_of_face_locations = face_recognition.batch_face_locations(
            frames, number_of_times_to_upsample=0
        )

        # Now let's list all the faces we found in all 128 frames
        for frame_number_in_batch, face_locations in enumerate(batch_of_face_locations):
            number_of_faces_in_frame = len(face_locations)

            frame_number = frame_count - 128 + frame_number_in_batch
            print(
                "I found {} face(s) in frame #{}.".format(
                    number_of_faces_in_frame, frame_number
                )
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
