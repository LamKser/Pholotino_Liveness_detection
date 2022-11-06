import pandas as pd
import os
import sys
import cv2
import tqdm
from cut_frame import cutFrame
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.abspath(ROOT))  # relative
WORK_DIR = os.path.dirname(ROOT)
sys.path.insert(0, WORK_DIR)


class classifyObject:
    def __init__(self, file_label):
        self.data_frame = pd.read_csv(os.path.join(WORK_DIR, 'datasets/train/' + file_label))
        # cutFrame.__init__(self, os.path.join(WORK_DIR, 'datasets/train/' + folder_video), 5)
        # self.frame, self.name_file_video, self.fps = cutFrame.__call__(self)
        # for file_video in os.listdir(folder_video):
        #     self.name_file_video = os.path.splitext(file_video)[0]
        #     # self.folder_save = os.path.join(WORK_DIR, os.path.splitext(file_video)[0])
        #     cutFrame.__init__(self, os.path.join(folder_video, file_video), 5)

    def __call__(self, folder_video, *args, **kwargs):
        self.data_frame = self.data_frame.to_dict('list')
        for file in tqdm.tqdm(list(self.data_frame['fname']), total=len(list(self.data_frame['fname']))):
            # if os.path.splitext(file)[0] == self.name_file_video:
            cuttingFrame = cutFrame(os.path.join(WORK_DIR, 'datasets/train/' + folder_video), file, 5)
            folder_save = os.path.join(WORK_DIR, 'train/' + str(
                self.data_frame['liveness_score'][list(self.data_frame['fname']).index(file)]) + '/' +
                                       os.path.splitext(file)[0])
            if not os.path.exists(folder_save):
                os.makedirs(folder_save)
            cuttingFrame(folder_save)
            # cutFrame.__call__(self, folder_save)
            # cv2.imwrite(os.path.join(WORK_DIR, folder_save + '/' + str(self.fps) + '.jpg'), self.frame)


if __name__ == '__main__':
    folderVideo = 'videos'
    fileLabel = 'label.csv'
    classifyFaceVideo = classifyObject(fileLabel)
    classifyFaceVideo(folderVideo)
