import cv2
import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.abspath(ROOT))  # relative
WORK_DIR = os.path.dirname(ROOT)
sys.path.insert(0, WORK_DIR)


class cutFrame:
    def __init__(self, folder_video, video_file, fps):
        self.fps = fps
        # self.folder_video = folder_video
        self.video = cv2.VideoCapture(os.path.join(folder_video, video_file))

    def __call__(self, folder_save, *args, **kwargs):
        fps = 0

        while self.video.isOpened():
            ret, frame = self.video.read()
            if not ret:
                break
            else:
                if fps % self.fps == 0:
                    cv2.imwrite(os.path.join(WORK_DIR, folder_save + '/' + str(fps) + '.jpg'), frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            fps = fps + 1
        self.video.release()
        cv2.destroyAllWindows()
