[![Zalo](https://img.shields.io/badge/zalo-0068FF?style=for-the-badge&logo=Zalo&logoColor=white)][1]
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

[1]: https://challenge.zalo.ai/#intro
# Liveness detection

## 1. Authors

- [Dinh Hoang Lam](https://github.com/LamKser)
- [Huynh Nhat Quoc Bao](https://github.com/baohnq)
- [Pham Minh Long](https://github.com/Syun1208)

## 2. About the project

We implement for the Zalo AI Challenge 2022 (Liveness detection)

## 3. Dataset

Zalo provide 3 types of dataset:
* `Training set`: **1168** videos of faces with facemask, in which **598** are real and **570** are fake. (0 = Fake, 1 = Real)
* `Public test`: **350** videos of faces with facemask, without label file.
* `Public test 2`: **486** videos of faces with facemask, without label file.
* `Private test`: **1253** videos of faces with facemask, without label file.

## 4. Run the project
* **Data preparation**

    Run `cut_frame.py` for cutting frames from training video set
    ```Python
    python cut_frame.py --folder-video <Input folder video path> --label-csv <Input label csv> --fps <Input fps>
    ```
    * **Note:** 
        
        * Your video folder must follow

            ```
            videos
            ├── 1.mp4
            ├── 2.mp4
            ├── ...
            └── xxx.mp4
            ```  

        * And your label csv must have 2 columns

            ```
            fname: file video name
            liveness_score: label (0 & 1)
            ```

* **Training**

    ```Python
    python main.py --train-path <video folder> --save-path <path for saving weight> --weight-file <weight name> --mode train
    ```
* **Test**

