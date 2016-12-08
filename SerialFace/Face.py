import cv2
import os
import numpy as np

from PIL import Image

import picamera.array
from picamera import PiCamera


class Face(object):

    training_count = 5
    threshold = 30

    def __init__(self, casc_path, path="./passwords", camera_port=0):
        self.path = path
        self._cascade = cv2.CascadeClassifier(casc_path)
        self._port = camera_port

    def __del__(self):
        cv2.destroyAllWindows()

    def _capture_image(self):
        """
        Throw away frames so we can let the camera adjust
        :return: list(list())
        """
        with PiCamera() as camera:
            with picamera.array.PiRGBArray(camera) as stream:
                camera.resolution = (640, 480)
                camera.capture(stream, 'bgr', use_video_port=True)
                return cv2.cv.cvtColor(stream.array, cv2.COLOR_BGR2GRAY)

    def _get_faces_and_frames(self):
        frame = self._capture_image()

        faces = self._cascade.detectMultiScale(
            frame,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        return faces, frame

    def _get_training_faf(self):
        """
        :yield: faces, frame
        Gets all images required for training.

        * Note
            Won't stop getting images unless there is only one face per image.
        """
        count = 0
        error_count = 0
        while count < self.training_count:  # Ensures that we get at least self.training_count images
            error_count += 1
            faces, frame = self._get_faces_and_frames()
            if len(faces) == 1:
                yield faces, frame
                count += 1
            elif error_count >= 10:
                break

    def can_unlock(self):
        """
        Will return false under the following conditions:
            1. More than one face in the image
            2. No images in password file
            3. Face is not recognized
        :return: True if face is recognized False if face is not recognized
        """

        face, frame = self._get_faces_and_frames()

        # Don't allow more than 1 face in the image
        if len(face) != 1:
            return False

        x, y, w, h = face[0]

        face = frame[y: y + h, x: x + w]

        recognizer = cv2.face.createLBPHFaceRecognizer()

        paths = [os.path.join(self.path, f) for f in os.listdir(self.path) if f.endswith("bmp")]

        if not paths:
            return False  # Return since there are no images saved as a password

        # images will contains face images
        images = []
        # labels will contains the label that is assigned to the image
        labels = []
        nbr = 0

        for image_path in paths:
            # Read the image
            image_pil = Image.open(image_path)
            # Convert the image format into numpy array
            image = np.array(image_pil, 'uint8')

            images.append(image)
            labels.append(nbr)
            nbr += 1

        cv2.destroyAllWindows()
        # Perform the tranining
        recognizer.train(images, np.array(labels))

        nbr_predicted, conf = recognizer.predict(face)

        if conf < self.threshold:
            return True

        return False

    def new_pass(self):
        count = 0
        for face, frame in self._get_training_faf():
            filename = "".join(["passwords/", str(count), ".bmp"])

            x, y, w, h = face[0]

            frame = frame[y: y + h, x: x + w]
            count += 1
            cv2.imwrite(filename, frame)

    def secure_new_pass(self):
        if self.can_unlock():
            self.new_pass()