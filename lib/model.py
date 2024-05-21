from __future__ import annotations


import numpy as np


class AnalysisData:
    # 웹, 앱 구분
    is_free: bool

    # 포즈 아이디
    pose_id: int

    # 스켈레톤 데이터 구성
    skeletonDatas: list[SkeletonData]

    # 섹션 리포트 배열 넣는곳
    sectionReportArray: list[SectionReport]

    # 스토리지 서버 스키마 대상 : 여기서는 초기화때 입력하지 않음

    # # 분석결과를 바탕으로 답변한 코멘트들
    # answer_data: list[str]

    # # 서버 아이피
    # server_ip: str

    # # SettingJson에 기입된 모델 name(아이디로 활용)
    # model_id: str

    @staticmethod
    def fromJsonString(jsonData):
        cls = AnalysisData()
        cls.skeletonDatas = [SkeletonData.fromJson(j) for j in jsonData["skeletonDatas"]]

        return cls

    def UpdateSectionReport(self, sectionReportArray: list[SectionReport]):
        """결과데이터 업데이트"""
        self.sectionReportArray = sectionReportArray

    def ToReportJson(self):
        jsonArray = [report.toJson() for report in self.sectionReportArray]
        return jsonArray

    def ToStorageJson(self, model_id, answer_data, s3_uuid, pose_id) -> dict:
        skeletonDatas = [skeleton.toJson() for skeleton in self.skeletonDatas]
        sectionReportArray = [report.toJson() for report in self.sectionReportArray]
        return {
            "skeletonDatas": skeletonDatas,
            "sectionReportArray": sectionReportArray,
            "s3_uuid": s3_uuid,
            "pose_id": pose_id,
            "model_id": model_id,
            "answer_data": answer_data,
        }


class SkeletonData:
    frameTime: int
    skeletons: list[Skeleton]

    @staticmethod
    def fromJson(jsonData):
        cls = SkeletonData()
        cls.skeletons = [Skeleton.fromJson(j) for j in jsonData["skeletons"]]

        return cls

    def toJson(self) -> dict:
        return {
            "frameTime": self.frameTime,
            "skeletons": [skeleton.toJson() for skeleton in self.skeletons],
        }

    def toArray33(self, platform: angleModule.PosePlatform) -> np.ndarray:
        """33개 데이터로 변환"""
        return np.array([skeleton.Array3(platform) for skeleton in self.skeletons],dtype=np.float32)


class Skeleton:
    x: float
    y: float
    z: float

    cx: float = None
    """보정된 x값"""
    cy: float = None
    """보정된 y값"""
    cz: float = None
    """보정된 z값"""

    @staticmethod
    def fromJson(jsonData):
        cls = Skeleton()
        cls.x = jsonData["x"]
        cls.y = jsonData["y"]
        cls.z = jsonData["z"]
        return cls

    # 보정된 데이터 반영하는 곳
    def UpdateCorrected(self, CV: np.NDArray, platform: angleModule.PosePlatform):
        if platform == angleModule.PosePlatform.pc:
            self.cx = self.x
            self.cy = self.y
            self.cz = CV[2]
        else:
            self.cx = self.x
            self.cy = self.y
            self.cz = CV[2] * 2.5

    def Array3(self, platform: angleModule.PosePlatform):
        if platform == angleModule.PosePlatform.pc:
            """PC인 경우에는 화면 비율에 맞게 1.8(16:9)을 곱해줘야함 X값에"""
            return np.array([self.x * 1.8, self.y, self.z], dtype=np.float32)
        else:
            """모바일인 경우에는 원본 데이터에는 2.5를 나눠줘야함"""
            return np.array([self.x, self.y, self.z / 2.5], dtype=np.float32)

    def toJson(self):
        return {"x": self.x, "y": self.y, "z": self.z}
    
    def GetTupleOriginalXY(self):
        return (self.x,self.y,0)



class Section:
    anglesArray: list[list[int]]
    """해당 섹션을 구성하는 각도 데이터 배열"""

    sMilliseconds: int
    """섹션 시작 시간 : 단위 ms """

    eMilliseconds: int
    """섹션 끝난 시간 : 단위 ms """

    predArray: list[float]
    """분석 결과 값"""

    def __init__(
        self,
        sMilliseconds: int,
        eMilliseconds: int,
        anglesArray: list[list[int]],
        predArray: list[float],
    ):
        self.anglesArray = anglesArray
        self.sMilliseconds = sMilliseconds
        self.eMilliseconds = eMilliseconds
        self.predArray = predArray

    @staticmethod
    def Create(
        anglesArray: list[list[int]],
        indexArray: list[int],
        predArray: list[float],
        windowSize: int,
    ):
        """추출한 범주 기반으로 섹션 생성, indexArray는 예측결과의 인덱스이기에 각도 데이터에 접근시에는 윈도우 크기에 영향을 받음"""

        addWindowSize: int = windowSize - 1

        # 윈도우 크기 반영해서 진행 : len(predArray)<len(angleArray) 이기에 오차는 addWindowSize만큼 차이남
        cls = Section(
            sMilliseconds=(indexArray[0] + addWindowSize)
            * 100,  # 시작하고 long window -1 크기만큼 더한후에 시간값 100곱함
            eMilliseconds=(indexArray[-1] + addWindowSize)
            * 100,  # 끝나고 long window -1 크기만큼 더한후에 시간값 100곱함
            anglesArray=anglesArray[indexArray[0] : indexArray[-1] + addWindowSize],
            predArray=predArray[indexArray[0] : indexArray[-1]],
        )

        return cls

    def GetTakenTime(self):
        if len(self.anglesArray) == 0:
            return 0

        frameCount: int = len(self.anglesArray[0])
        frameCount *= 100  # 프레임당 소요시간 100ms로 잡고 단위로 변경
        return frameCount
    


class SectionReport:
    # 1회 운동 데이터 분석 결과
    angleReports: list[AngleReport]
    """각도별 결과리포트"""
    timeDiff: float
    """정답데이터와 1회 운동시 소요된 시간 차이값 표기 (0이 좋음)"""

    def __init__(self, angleReports, timeDiff):
        self.angleReports = angleReports
        self.timeDiff = timeDiff

    def toJson(self) -> dict:
        return {
            "angleReports": [angleReport.toJson() for angleReport in self.angleReports],
            "time": self.timeDiff,
        }


class AngleReport:
    # 각도별 분석 결과
    angleIndex: int
    """각도 번호"""
    dtw: float
    """유사도 체크 (0이 제일 좋고 값이 클수록 차이가 심하다는것을 의미함"""
    minDiff: float
    """유저의 최소값과 트레이너의 최소값간의 차이점 도출(0이 좋고, 값이 크거나 작을수록 심함)"""
    maxDiff: float
    """유저의 최대값과 트레이너의 최대값간의 차이점 도출(0이 좋고, 값이 크거나 작을수록 심함)"""
    min: float
    """최소 가동범위"""
    max: float
    """최대 가동범위"""

    def __init__(self, **k):
        self.angleIndex = k["angleIndex"]
        self.dtw = k["dtw"]
        self.minDiff = k["minDiff"]
        self.maxDiff = k["maxDiff"]
        self.min = k["min"]
        self.max = k["max"]

    def toJson(self) -> dict:
        return {
            "angleIndex": self.angleIndex,
            "dtw": self.dtw,
            "minDiff": self.minDiff,
            "maxDiff": self.maxDiff,
            "min": self.min,
            "max": self.max,
        }


class AnswerData:
    name: str
    des: str

    anglesArray: dict
    """정답 각도 데이터가 들어가있는구조"""

    takenTime: int
    """단위는 ms"""

    @staticmethod
    def fromJson(jsonData):
        cls = AnswerData()
        cls.name = jsonData["pose_uuid"]
        cls.des = jsonData["des"]
        cls.anglesArray = jsonData["anglesArray"]
        cls.takenTime = jsonData["takenTime"]

        return cls

    def getIndexArray(self) -> list[int]:
        """인덱스 데이터 불러오기"""
        return [int(key) for key in self.anglesArray]

    def getAngleArray(self) -> list[list[int]]:
        """각도 데이터 불러오기"""
        return [self.anglesArray[key] for key in self.anglesArray]


import lib.anglelib as angleModule  # 각도 추출 알고리즘
