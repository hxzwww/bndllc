import cv2
from yolov5 import YOLOv5


class Detector:
    """
    A class for detecting objects, specifically persons,
    in video files using the YOLOv5 model.

    Attributes:
        model (YOLOv5): YOLOv5s model for object detection.
        cap (cv2.VideoCapture): Video capture object for reading video files.
        fps (float): Frames per second of the input video.
        frame_height (int): Height of the video frames.
        frame_width (int): Width of the video frames.
        frames (list): List of processed video frames.
    """

    def __init__(self):
        """
        Initializes the Detector object with the YOLOv5 model
        and sets up attributes for video processing.
        """
        self.model = YOLOv5('yolov5s.pt')
        self.cap = None
        self.fps = None
        self.frame_height = None
        self.frame_width = None
        self.frames = None

    def load(self, input_file):
        """
        Loads a video file for processing.

        Parameters:
            input_file (str): Path to the input video file.

        Returns:
            None
        """
        self.cap = cv2.VideoCapture(input_file)

    def save(self, output_file):
        """
        Saves the processed video frames to an output file.

        Parameters:
            output_file (str): Path to the output video file.

        Returns:
            None
        """
        output_video = cv2.VideoWriter(
            output_file, cv2.VideoWriter_fourcc(*'mp4v'),
            self.fps, (self.frame_width, self.frame_height)
        )
        for frame in self.frames:
            output_video.write(frame)
        output_video.release()

    def run(self):
        """
        Processes the loaded video, performing object detection
        on each frame and saving the results.

        Returns:
            None
        """
        if self.cap is None:
            print('Load file before running.')
            return

        self.frames = []
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.__process()

        self.cap.release()

    def __process(self):
        """
        Internal method to perform object detection on video frames.
        Detects objects using YOLOv5, draws bounding boxes around persons,
        and adds text labels.

        Returns:
            None
        """
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            results = self.model.predict(frame)

            for det in results.xyxy[0]:
                xyxy = det[:4].cpu().numpy().astype(int)
                conf = det[4].cpu().numpy()
                cls = int(det[5].cpu().numpy())
                label = f'{results.names[cls]} {conf:.2f}'
                if label.split()[0] == 'person':
                    cv2.rectangle(
                        frame, (xyxy[0], xyxy[1]),
                        (xyxy[2], xyxy[3]), (0, 255, 0), 2
                    )
                    cv2.putText(
                        frame, label, (xyxy[0], xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2
                    )
            self.frames.append(frame)
