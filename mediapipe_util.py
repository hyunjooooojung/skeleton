from time import time
import random

from datetime import datetime

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2


from lib.model import AnalysisData
from lib.anglelib import Calculate, PosePlatform


class MediapipeUtil:
    def __init__(self) -> None:
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
        )

    async def draw_landmarks_on_image(self, rgb_image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        
        if rgb_image.shape[2] == 4:  # 4개의 채널을 가진 경우 (RGBA)
            # Alpha 채널을 제거하고 BGR 형식으로 변환
            rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        
        annotated_image = np.copy(rgb_image)

        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend(
                [
                    landmark_pb2.NormalizedLandmark(
                        x=landmark.x, y=landmark.y, z=landmark.z
                    )
                    for landmark in pose_landmarks
                ]
            )
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style(),
            )
        return annotated_image

    async def save_landmarks_image(self, filename):
        base_options = python.BaseOptions(model_asset_path="pose_landmarker_lite.task")
        options = vision.PoseLandmarkerOptions(
            base_options=base_options, output_segmentation_masks=True
        )
        detector = vision.PoseLandmarker.create_from_options(options)

        image = mp.Image.create_from_file(f"tmp/{filename}")

        detection_result = detector.detect(image)

        annotated_image = await self.draw_landmarks_on_image(
            image.numpy_view(), detection_result
        )
        img_cv = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"tmp/out_{filename}", img_cv)

    async def extract_landmark_from_memory(self,contents,filename):
        # BytesIO 객체로 이미지를 numpy array 형태로 변환
        nparr = np.frombuffer(contents, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Mediapipe 설정
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

        # 이미지 처리
        results = pose.process(img_np)
        
        # 랜드마크 그리기
        annotated_image = img_np.copy()
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=results.pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
            )

        # 결과 이미지 저장
        cv2.imwrite(f'tmp/output_{filename}', annotated_image)
        
        return {"filename": filename, "status": "File processed and landmarks drawn"}

    async def extract_landmark_from_image(self, filename):
        image = cv2.imread(f"tmp/{filename}")

        if image is None:
            raise ValueError("Image not loaded. Please check the path and try again.")

        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        landmarks_data = []
        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                landmarks_data.append(
                    {
                        "landmark": idx,
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z,
                    }
                )

        return {"skeletonDatas": {"skeletons": landmarks_data}}
    
    def extract_landmark_from_uploaded_memory(self, contents):
        # BytesIO 객체로 이미지를 numpy array 형태로 변환
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Image not loaded. Please check the file and try again.")

        # BGR에서 RGB로 변환하여 Mediapipe에 적용
        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        landmarks_data = []
        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                landmarks_data.append(
                    {
                        "landmark": idx,
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z,
                    }
                )

        return {"skeletonDatas": {"skeletons": landmarks_data}}
    
    async def extract_landmark_from_image_angle(self, filename):
        image = cv2.imread(f"tmp/{filename}")

        if image is None:
            raise ValueError("Image not loaded. Please check the path and try again.")

        results = self.pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        landmarks_data = []
        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                landmarks_data.append(
                    {
                        "landmark": idx,
                        "x": landmark.x,
                        "y": landmark.y,
                        "z": landmark.z,
                    }
                )

        return {"skeletonDatas": [{"skeletons": landmarks_data}]}

    async def extract_landmark_from_video(self, filename):
        cap = cv2.VideoCapture(f"tmp/{filename}")
        all_landmarks = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = self.pose.process(frame_rgb)

            if results.pose_landmarks:
                frame_landmarks = []
                if frame_count == 0:
                    for idx, landmark in enumerate(results.pose_landmarks.landmark):
                        frame_landmarks.append(
                            {
                                "landmark": idx,
                                "x": landmark.x,
                                "y": landmark.y,
                                "z": landmark.z,
                            }
                        )
                    all_landmarks.append({"skeletons": frame_landmarks})
                    frame_count = 2
                else:
                    frame_count -= 1

        cap.release()

        return {"skeletonDatas": all_landmarks}

    async def extract_angles_from_landmark(self, landmark_data):
        start = time()
                # 유닉스 타임스탬프를 datetime 객체로 변환
        start_datetime = datetime.fromtimestamp(start)

        # datetime 객체를 사람이 읽을 수 있는 문자열로 포맷팅
        formatted_time_s = start_datetime.strftime('%Y-%m-%d %H:%M:%S')
        print("보정시작: ", start,formatted_time_s)
        analysisData: AnalysisData = AnalysisData.fromJsonString(landmark_data)

        array33List: list[np.ndarray] = [
            skeletonData.toArray33(PosePlatform.pc)
            for skeletonData in analysisData.skeletonDatas
        ]
        array33NP = np.array(array33List, dtype=np.float32)
        
        anglesArray: list[list[int]] = Calculate(array33NP, isFixed=True)
        end = time()
        print("anglesArray length:",len(anglesArray)*16)
        end_datetime = datetime.fromtimestamp(end)

        # datetime 객체를 사람이 읽을 수 있는 문자열로 포맷팅
        formatted_time_e = end_datetime.strftime('%Y-%m-%d %H:%M:%S')
        print("보정끝: ", end,formatted_time_e)
        print("보정시간(s): ", end - start)

        return anglesArray

 

        return anglesArray
