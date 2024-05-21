from __future__ import annotations
import numpy as np
import math
from enum import IntEnum
from numba import njit


class PosePlatform(IntEnum):
    mobile = 1
    pc = 2


ANGLE_NAMES = [
    "왼쪽 팔꿈치",
    "오른쪽 팔꿈치",
    "왼쪽 무릎",
    "오른쪽 무릎",
    "왼팔 어깨 기준 각도",
    "왼팔 전방 기준 각도",
    "오른팔 어깨 기준 각도",
    "오른팔 전방 기준 각도",
    "허리 XY평면 돌리기 각도",
    "허리 XZ 평면 돌리기 각도",
    "좌우어깨 기울기 각도",
    "왼쪽다리 전방 기준 각도",
    "왼쪽다리 전방 기준 각도",
    "왼쪽다리 골반 기준 각도",
    "오른쪽다리 골반 기준 각도",
    "허리 굽힘 각도",
]


#@njit
def Calculate(array33List, isFixed: bool):
    count: int = len(array33List)
    return [
        UnitCalculate(isFixed, idx, array33List[idx], array33List[idx - 1] if idx != 0 else None) for idx in range(count)
    ]


#@njit
def UnitCalculate(isFixed: bool, idx: int, array33, array33_pre):
    """"""
    if len(array33) > 0:
        if isFixed:
            # 보정 개선
            Correct(idx, array33, array33_pre)
            # 길이 보정 진행
            Precessing(idx, array33)
        angles: list[int] = GetAngleList(array33)
        return angles
    return None


#@njit
def GetAngleList(array33: np.ndarray):
    """각도 리스트 불러오기 : 15개"""
    a0 = LeftElbow(array33)
    a1 = RightElbow(array33)
    a2 = LeftKnee(array33)
    a3 = RightKnee(array33)
    a4 = LeftShoulder(array33)  # 11에서 변경됨
    a5 = LeftShoulderFront(array33)  # 12에서 변경됨
    a6 = RightShoulder(array33)  # 11에서 변경됨
    a7 = RightShoulderFront(array33)  # 12에서 변경됨
    a8 = WaistXY(array33)
    a9 = WaistXZ(array33)
    a10 = Shoulder(array33)
    a11 = LeftLegFront(array33)
    a12 = RightLegFront(array33)
    a13 = LeftLeg(array33)
    a14 = RightLeg(array33)
    a15 = WaistFolding(array33)  # 14에서 추가됨

    return [a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15]


############################################################################################################
############################################################################################################
############################################################################################################


# 몸통 전면 벡터
#@njit
def GetVerticalVector(array33):
    """몸통 정면 벡터"""
    v1 = array33[11]
    v2 = array33[12]
    p23 = array33[23]
    p24 = array33[24]

    v3 = (p23 + p24) / 2

    pq = v2 - v1
    pr = v3 - v1

    return GetNormalVector(pq, pr)


#@njit
def GetLowBodyVerticalVector(array33):
    """하체 정면 벡터 : q->p->r 왼손 엄지방향"""
    v1 = array33[26]  # p
    v2 = array33[25]  # q
    p23 = array33[23]
    p24 = array33[24]

    v3 = (p23 + p24) / 2  # r

    pq = v2 - v1
    pr = v3 - v1

    return GetNormalVector(pq, pr)


#@njit
def GetNormalVector(pq, pr):
    """평면 벡터 생성"""

    x = pq[1] * pr[2] - pq[2] * pr[1]
    y = pq[2] * pr[0] - pq[0] * pr[2]
    z = pq[0] * pr[1] - pq[1] * pr[0]

    return np.array([x, y, z], dtype=np.float32)


# 몸통 벡터 생성
#@njit
def GetBodyVector(array33):
    """N(11-12) ,M(23-24)를 지나는 벡터로 방향은 머리를 향함"""
    p11 = array33[11]
    p12 = array33[12]
    p23 = array33[23]
    p24 = array33[24]

    tm = (p11 + p12) / 2
    bm = (p23 + p24) / 2

    return tm - bm


# 두 백터간의 각도 측정
#@njit
def GetVectorAngle(ab, cb):
    dot = np.dot(ab, cb)
    abl = np.linalg.norm(ab)
    cbl = np.linalg.norm(cb)
    value = dot / (abl * cbl)

    if value > 1:
        value = 1.0
    if value < -1:
        value = -1
    rad = np.arccos(value)
    return np.float32(np.degrees(rad))


############################################################################################################
############################################################################################################
############################################################################################################
#@njit
def LeftElbow(array33):
    """
    ## 0. 왼쪽 팔꿈치
    ### 범위 0~180도
    * 0 :다 굽힌경우(실제 45도까지밖에 안됨
    * 180 : 다 핀경우"""

    # isFixed = True : 보정된 값으로 하지 않겠다 (현재 보정 유효도 낮아서)
    a = array33[11]
    b = array33[13]
    c = array33[15]

    ba = a - b
    bc = c - b

    # 각도 추출
    # angle=Vector(ba).angle_between(bc)
    # angle=round(math.degrees(angle))

    angle = GetVectorAngle(ba, bc)

    return angle


#@njit
def RightElbow(array33):
    """
    ## 1. 오론쪽 팔꿈치
    ### 범위 0~180도
    * 0 :다 굽힌경우(실제 45도까지밖에 안됨
    * 180 : 다 핀경우"""

    # isFixed = True : 보정된 값으로 하지 않겠다 (현재 보정 유효도 낮아서)
    a = array33[12]
    b = array33[14]
    c = array33[16]

    ba = a - b
    bc = c - b

    # 각도 추출
    # angle=Vector(ba).angle_between(bc)
    # angle=round(math.degrees(angle))

    angle = GetVectorAngle(ba, bc)

    return angle


#@njit
def LeftKnee(array33):
    """
    ## 2. 왼쪽 무릎
    ### 범위 0~180도
    * 0 :다 굽힌경우(실제 45도까지밖에 안됨
    * 180 : 다 핀경우"""
    a = array33[23]
    b = array33[25]
    c = array33[27]

    ba = a - b
    bc = c - b

    # 각도 추출
    # angle=Vector(ba).angle_between(bc)
    # angle=round(math.degrees(angle))

    angle = GetVectorAngle(ba, bc)

    return angle


#@njit
def RightKnee(array33):
    """
    ## 3. 오론쪽 무릎
    ### 범위 0~180도
    * 0 :다 굽힌경우(실제 45도까지밖에 안됨
    * 180 : 다 핀경우"""
    a = array33[24]
    b = array33[26]
    c = array33[28]

    ba = a - b
    bc = c - b

    # 각도 추출
    angle = GetVectorAngle(ba, bc)

    return angle


#@njit
def LeftShoulder(array33):
    """
    ## 4. 왼팔 어깨 드는 정도 기준 각도
    ### 범위 0~180도
    * 0 :팔을 붙인 경우
    * 180 : 다 핀경우
    * 180 이상 : 손을 머리쪽으로 뻣은 상태에서 몸방향으로 더 뻣은경우
    """

    a = array33[23]
    b = array33[11]
    c = array33[13]

    ba = a - b
    bc = c - b

    # 각도 추출
    angle = GetVectorAngle(ba, bc)

    # 180 이상 판별 데이터 생성
    d = array33[12]
    bd = d - b
    bc = c - b

    angle_check = GetVectorAngle(bd, bc)

    # 180도 이상 판정 기준
    if angle_check < 90 and angle > 150:
        angle = 360 - angle

    return angle


#@njit
def LeftShoulderFront(array33):
    """
    17버전에서 변경됨
    """
    # 평면 생성 포인트 초기화
    mt = (array33[11] + array33[12]) / 2
    mb = (array33[23] + array33[24]) / 2
    planeNormal = mt - mb

    # 점벡터 평면에 투영
    pp13 = PlaneProjectionNormal(mt, planeNormal, array33[13])
    pp11 = PlaneProjectionNormal(mt, planeNormal, array33[11])
    pp12 = PlaneProjectionNormal(mt, planeNormal, array33[12])

    # 조건부 삽입 : 너무 값이 작은 경우 대응
    pl1113 = Distance(pp13, pp11)
    pl1112 = Distance(pp12, pp11)
    if pl1112 * 0.3 > pl1113:
        return 120.0

    # 각도 추출
    angle = GetVectorAngle(pp13 - pp11, pp12 - pp11)  # 각도 추출

    # 상체 정면 벡터 : 몸통 앞방향
    upper_body_vector = GetVerticalVector(array33)

    # 예각 둔각 처리
    sign_angle = GetVectorAngle(upper_body_vector, pp13 - pp11)
    angle = angle if sign_angle < 90 else 360 - angle

    # 부호 반영
    return angle


#@njit
def RightShoulder(array33):
    """
    ## 6. 오른팔 어깨 기준 각도
    ### 범위 0~180도
    * 0 :다 굽힌경우(실제 45도까지밖에 안됨
    * 180 : 다 핀경우"""

    a = array33[24]
    b = array33[12]
    c = array33[14]

    ba = a - b
    bc = c - b

    # 각도 추출
    angle = GetVectorAngle(ba, bc)

    # 180 이상 판별 데이터 생성
    d = array33[11]
    bd = d - b
    bc = c - b

    angle_check = GetVectorAngle(bd, bc)

    # 180도 이상 판정 기준
    if angle_check < 90 and angle > 150:
        angle = 360 - angle

    return angle


#@njit
def RightShoulderFront(array33):
    """
    17버전에서 변경됨
    """
    # 평면 생성 포인트 초기화
    mt = (array33[11] + array33[12]) / 2
    mb = (array33[23] + array33[24]) / 2
    planeNormal = mt - mb

    # 점벡터 평면에 투영
    pp14 = PlaneProjectionNormal(mt, planeNormal, array33[14])
    pp11 = PlaneProjectionNormal(mt, planeNormal, array33[11])
    pp12 = PlaneProjectionNormal(mt, planeNormal, array33[12])

    # 조건부 삽입 : 너무 값이 작은 경우 대응
    pl1214 = Distance(pp12, pp14)
    pl1112 = Distance(pp11, pp12)
    if pl1112 * 0.3 > pl1214:
        return 120.0

    # 각도 추출
    angle = GetVectorAngle(pp14 - pp12, pp11 - pp12)  # 각도 추출

    # 상체 정면 벡터 : 몸통 앞방향
    upper_body_vector = GetVerticalVector(array33)

    # 예각 둔각 처리
    sign_angle = GetVectorAngle(upper_body_vector, pp14 - pp12)
    angle = angle if sign_angle < 90 else 360 - angle

    # 부호 반영
    return angle


#@njit
def WaistXY(array33):
    """
    ## 8. 허리XY평면 돌리기 각도
    ### 범위 0~180도
    * 90 :서있는 경우
    * 0 :왼쪽으로 완전 기울인 경우
    * 180 : 오른쪽으로 완전 기울인 경우
    """

    # 몸통벡터 : 머리방향
    body_vector = GetBodyVector(array33)

    # 23,24 포인트 추출
    letf_hip_point = array33[23]
    right_hip_point = array33[24]

    # 골반 벡터 추출 : 왼쪽 방향
    v2423 = letf_hip_point - right_hip_point

    # 각도 추출
    # angle=Vector(body_vector).angle_between(v2423)
    # angle=round(math.degrees(angle))
    angle = GetVectorAngle(body_vector, v2423)

    return angle


#@njit
def WaistXZ(array33):
    """
    ## 9. 허리XZ평면 돌리기 각도
    ### 범위 -45~225도
    * 90 :서있는 경우
    * 0 :왼쪽으로 완전 기울인 경우
    * 180 : 오른쪽으로 완전 기울인 경우
    * -45: 왼쪽으로 끝까지 돌려서 뒤를 바라본경우
    * 225: 오른쪽으로 끝까지 돌려서 뒤로 바라본 경우
    """
    # 정면 벡터와 23,24간의 각도
    # 23,24 와 11,12간의 각도가 둔각이면서 9번값이 예각인경우 경우 - , 23,24 와 11,12간의 각도가 둔각이면서 9번값이 둔각인 경우 360-9번값 을 입력함

    # 상체 정면 벡터 : 몸통 앞방향
    upper_body_vector = GetVerticalVector(array33)

    # 23,24 포인트 추출
    letf_hip_point = array33[23]
    right_hip_point = array33[24]

    # 골반 벡터 추출 : 왼쪽 방향
    v2423 = letf_hip_point - right_hip_point

    # 각도 추출
    # angle=Vector(v2423).angle_between(upper_body_vector)
    # angle=round(math.degrees(angle))
    angle = GetVectorAngle(v2423, upper_body_vector)

    # +/- 처리
    letf_shoulder_point = array33[11]
    right_shoulder_point = array33[12]
    v1211 = letf_shoulder_point - right_shoulder_point

    # sign_angle=Vector(v1211).angle_between(v2423)
    # sign_angle=round(math.degrees(sign_angle))
    sign_angle = GetVectorAngle(v1211, v2423)

    sign = 1 if sign_angle > 90 else -1

    # 부호 반영 (둔각인 경우에만)
    if sign == 1:
        if angle < 90:
            angle = -angle
        else:
            angle = 360 - angle

    return angle


#@njit
def Shoulder(array33):
    """
    ## 10. 좌우 어깨 기울기 각도
    ### 범위 45~135도
    * 45 : 왼쪽으로 끝까지 어깨를 기울인 경우
    * 90: 차렷인 경우
    * 135 : 오른쪽으로 끝까지 어깨를 기울인 경우
    """

    # 몸통벡터 : 머리방향
    body_vector = GetBodyVector(array33)

    # 11,12 포인트 추출
    letf_shoulder_point = array33[11]
    right_shoulder_point = array33[12]
    v1112 = right_shoulder_point - letf_shoulder_point

    # 각도 추출
    # angle=Vector(v1112).angle_between(body_vector)
    # angle=round(math.degrees(angle))
    angle = GetVectorAngle(v1112, body_vector)

    return angle


#@njit  # 테스트용
def LeftLegFront(array33):
    """
    ## 11. 왼쪽다리 전방 기준 각도
    ### 범위 -270~135도
    * -90: 일반 서있는 자세
    * 0: 앞발차기 자세
    * 135 : 다리를 어깨 넘어로 45도 더 보낸 요가자세인 경우 해당
    * -270: 다리를 뒤로 끝까지 넘긴 요가자세인경우 해당
    """
    # 중심 벡터와 정면 벡터를 통해 평면을 만든뒤 23,25 벡터를 투영하여 만든 벡터와 정면 벡터와의 각도(앞 + -)
    # 중심 벡터와 정면 벡터를 통해 평면을 만든뒤 23,25 벡터를 투영하여 만든 벡터와 중심 벡터와의 각도가  90도 미만(예각이면) -, 90도 이상(둔각이면)+를 입력함

    # 몸통벡터 : 머리방향
    body_vector = GetBodyVector(array33)

    # 상체 정면 벡터 : 몸통 앞방향
    upper_body_vector = GetVerticalVector(array33)

    # 투영 포인트 / 벡터 추출
    projected_point = PlaneProjectionNormal(array33[23], array33[23] - array33[24], array33[25])

    projected_vector = projected_point - array33[23]
    # +/- 처리 (B)
    sign_angle = GetVectorAngle(projected_vector, body_vector)

    # 각도 추출 (A)
    angle = GetVectorAngle(projected_vector, upper_body_vector)

    # 앞뒤 분류
    if angle < 90 and sign_angle < 90:
        # 둘다 예각인경우 -> -음수화
        angle = -angle

    elif angle > 90 and sign_angle < 90:
        # 메인 둔각(A) , 식별 예각(B)
        angle = 360 - angle

    return angle


#@njit  # 테스트용
def RightLegFront(array33):
    """
    ## 12. 오른쪽 다리 전방 기준 각도
    ### 범위 -270~135도
    * -90: 일반 서있는 자세
    * 0: 앞발차기 자세
    * 135 : 다리를 어깨 넘어로 45도 더 보낸 요가자세인 경우 해당
    * -270: 다리를 뒤로 끝까지 넘긴 요가자세인경우 해당
    """
    # 몸통벡터 : 머리방향
    body_vector = GetBodyVector(array33)

    # 상체 정면 벡터 : 몸통 앞방향
    upper_body_vector = GetVerticalVector(array33)

    # 투영 포인트 / 벡터 추출
    projected_point = PlaneProjectionNormal(array33[24], array33[24] - array33[23], array33[26])

    projected_vector = projected_point - array33[24]

    # +/- 처리 (B)
    sign_angle = GetVectorAngle(projected_vector, body_vector)

    # 각도 추출 (A)
    angle = GetVectorAngle(projected_vector, upper_body_vector)

    # 앞뒤 분류
    if angle < 90 and sign_angle < 90:
        # 둘다 예각인경우 -> -음수화
        angle = -angle

    elif angle > 90 and sign_angle < 90:
        # 메인 둔각(A) , 식별 예각(B)
        angle = 360 - angle

    return angle


#@njit
def LeftLeg(array33):
    """
    ## 13. 왼쪽다리 골반 기준 각도
    ### 범위 45~180도
    * 45 : 몸안쪽으로 다리를 모운경우
    * 90: 차렷일때
    * 180 : 몸 최대 바깥쪽으로 다리를 벌린경우
    """
    a = array33[24]
    b = array33[23]
    c = array33[25]

    ba = a - b
    bc = c - b

    # 각도 추출
    # angle=Vector(ba).angle_between(bc)
    # angle=round(math.degrees(angle))
    angle = GetVectorAngle(ba, bc)

    return angle


#@njit
def RightLeg(array33):
    """
    ## 14. 오른쪽 다리 골반 기준 각도
    ### 범위 45~180도
    * 45 : 몸안쪽으로 다리를 모운경우
    * 90: 차렷일때
    * 180 : 몸 최대 바깥쪽으로 다리를 벌린경우
    """
    a = array33[23]
    b = array33[24]
    c = array33[26]

    ba = a - b
    bc = c - b

    # 각도 추출
    # angle=Vector(ba).angle_between(bc)
    # angle=round(math.degrees(angle))
    angle = GetVectorAngle(ba, bc)

    return angle


#@njit
def WaistFolding(array33):
    """15. 허리 숙임 폴딩 각도"""
    # 상체 정면 벡터
    upper_body_vector = GetVerticalVector(array33)
    # 하체 정면 벡터
    low_body_vector = GetLowBodyVerticalVector(array33)

    # angle=Vector(low_body_vector).angle_between(upper_body_vector)
    # angle=round(math.degrees(angle))
    angle = GetVectorAngle(low_body_vector, upper_body_vector)

    return angle


#######################################################################################
# Length 파트
#######################################################################################


#@njit
def Precessing(
    idx: int,
    array33: np.ndarray,
    armCorrectRatio: float = 0.8,
    legCorrectRatio: float = 0.85,
):
    """양팔 양다리 끝지점 보정 진행
    - legCorrectRatio - >다리 길이 조정 비율 : 다리는 어떤 비율로 줄일지
    - armCorrectRatio -> 팔 길이 조정 비율 : 팔은 어떤 비율로 줄일지
    """

    # 팔쪽 보정 진행
    targetArmLengthLeft = GetTargetLength(array33[11], array33[13], ratio=armCorrectRatio)
    targetArmLengthRight = GetTargetLength(array33[12], array33[14], ratio=armCorrectRatio)

    # 팔쪽 보정 데이터 값 추출
    correctP15: np.ndarray = LengthCorrect(array33[13], array33[15], targetArmLengthLeft)
    correctP16: np.ndarray = LengthCorrect(array33[14], array33[16], targetArmLengthRight)

    # 팔쪽 길이 보정 완료
    array33[15] = correctP15
    array33[16] = correctP16
    # logger.info(f"  => correctP15:{correctP15},correctP16:{correctP16}") #@ 테스트 로그

    # 다리쪽 보정 진행
    targetLegLengthLeft = GetTargetLength(array33[23], array33[25], ratio=legCorrectRatio)
    targetLegLengthRight = GetTargetLength(array33[24], array33[26], ratio=legCorrectRatio)
    # logger.info(f"{idx}.Precessing -> Leg -> CompareNearScreen(True) -> targetArmLength:{targetArmLength}") #@ 테스트 로그
    # logger.info(f"{idx}.Precessing -> Leg -> CompareNearScreen(False) -> targetArmLength:{targetArmLength}") #@ 테스트 로그

    # 다리쪽 보정 데이터 값 추출
    correctP27: np.ndarray = LengthCorrect(array33[25], array33[27], targetLegLengthLeft)
    correctP28: np.ndarray = LengthCorrect(array33[26], array33[28], targetLegLengthRight)

    # 다리쪽 길이 보정 완료
    array33[27] = correctP27
    array33[28] = correctP28
    # logger.info(f"  => correctP27:{correctP27},correctP28:{correctP28}") #@ 테스트 로그


#@njit
def LengthCorrect(basePoint: np.ndarray, targetPoint: np.ndarray, targetLength: float) -> np.ndarray:
    """
    ### 길이 보정 함수
    두점간의 길이를 비교하여 XY 평면상에서 계산을 통해서 보정 진행
    - basePoint : 기준이되는 지점 (무릎 or 팔꿈치)
    - targetPoint : 변경하고자 하는 지점 (팔목 or 발목)
    - targetLength : base와 target간에 보정하고자 하는 목표 길이

    -> 반환 값은 변경된 targetPoint값
    """
    # # 분석대상 데이터 변경

    # z값이 모두 0인 base와  targetPoint값을 기반으로 길이 측정(xy 평면상의 길이) -> 2
    base2d: np.ndarray = array2D(basePoint)
    target2d: np.ndarray = array2D(targetPoint)
    length2d: float = GetLength(base2d, target2d)

    # 절대값을 기반으로 보정이 가능한지 체크, 0보다 작으면 제곱근 불가하기에 보정하지않고 넘김
    squareDiff: float = (targetLength * targetLength) - (length2d * length2d)
    if squareDiff < 0:
        targetPoint[2] = basePoint[2]
        return targetPoint

    # (3^2-2^2) 의 루트 값인 z변화량(절대값임) 추출 -> 4   (3은 targetLength)
    diffZ: float = math.sqrt((targetLength * targetLength) - (length2d * length2d))

    # 보정값 반영 방향 값
    plus: bool = True if targetPoint[2] - basePoint[2] > 0.0 else False

    # 보정값
    correctZ: float = basePoint[2] + diffZ if plus else basePoint[2] - diffZ

    # 보정값 반영
    targetPoint[2] = correctZ

    return targetPoint


#@njit
def GetLength(n1: np.ndarray, n2: np.ndarray):
    """두점간의 길이 추출"""
    # p1=Point(n1)
    # p2=Point(n2)
    # distance:float=p2.distance_point(p1) # 변하기 전 처리값

    distance: float = np.linalg.norm(n2 - n1)
    distance = distance if distance != None else 0

    return distance


#@njit
def array2D(n: np.ndarray) -> np.ndarray:
    """2차원 길이"""
    return np.array([n[0], n[1]])


#@njit
def CompareNearScreen(s1: np.ndarray, s2: np.ndarray) -> np.ndarray:
    """
    누가 화면에 가까운지 체크하는 곳
    - 작을수록 화면에 가까운 값
    """
    return s1[2] < s2[2]


#@njit
def GetTargetLength(rootPoint: np.ndarray, basePoint: np.ndarray, ratio: float):
    """보정하려는 길이 추출"""

    # 루트와 베이스 간의 거리 측정
    rootLength: float = GetLength(rootPoint, basePoint)

    return rootLength * ratio


############################################################################################################
# overchec 파트 : 외부 호출 함수
############################################################################################################
#@njit
def Correct(idx: int, array33: np.ndarray, array33_pre: np.ndarray):
    # 왼쪽 팔 보정 여부
    leftArmNeedCorrect: int = CheckLeftArm(idx, array33, array33[13])

    if leftArmNeedCorrect != 0:
        # 보정 진행
        # LEFT_ARM_ANGLES:list[int]=[11,13,15]
        OverCorrect(idx, array33, array33_pre, idxList=[11, 13, 15], isLeft=True)
        # print("\n")

    # 오른쪽 팔 보정 여부
    rightArmNeedCorrect: int = CheckRightArm(idx, array33, array33[14])

    if rightArmNeedCorrect != 0:
        # 보정 진행
        # RIGHT_ARM_ANGLES:list[int]=[12,14,16]
        OverCorrect(idx, array33, array33_pre, idxList=[12, 14, 16], isLeft=False)


############################################################################################################
# overchec 파트 : 내부 함수
############################################################################################################
#@njit
def CheckLeftArm(idx: int, array33: np.ndarray, v13: np.ndarray) -> int:
    """"""

    # 포인트 불러오기
    v11 = array33[11]
    v12 = array33[12]
    v15 = array33[15]

    v23 = array33[23]
    v24 = array33[24]

    # logger.info(f"{idx} Numpy : {v11},{v12},{v15},{v23},{v24}") #@ 테스트 로그

    m1112: np.ndarray = (v11 + v12) / 2
    m2324: np.ndarray = (v23 + v24) / 2
    bodyVector = m1112 - m2324

    # 어깨 근육 방향 벡터 추출 유효성 체크 : 팔을 핀 경우에는 어깨근육 벡터가 상당히 모호해짐
    armAngle: float = GetVectorAngle(v11 - v13, v15 - v13)

    if armAngle > 175:
        # print("펴져있어서 어깨근육으로 판별 불가하여 정상으로 반환")
        # logger.info(f"{idx}.CheckLeftArm -> armAngle > 175 -> return 0") #@ 테스트 로그
        return 0

    # 어깨 근육 방향 벡터 추출
    shoulderVector: np.ndarray = GetNormalVector(v11 - v13, v15 - v13)
    shoulderPoint: np.ndarray = shoulderVector + v11

    # 머리위로 점
    headPoint = bodyVector + v11
    # logger.info(f"{idx}.CheckLeftArm -> v11-v13,v15-v13 :{v11-v13,v15-v13},shoulderVector:{shoulderVector},shoulderPoint:{shoulderPoint},headPoint:{headPoint},bodyVector:{bodyVector}") #@ 테스트 로그

    # 측면 평면 : 머리 쪽 투영해서 예각인 경우에만 오케이
    sideAngle: float = SideShoulderPlaneAngle(
        mainShoulder=v11,
        otherShoulder=v12,
        shoulderPoint=shoulderPoint,
        headPoint=headPoint,
    )

    # 정면 평면 : 1112와 각도값이 예각이면서 1112m 과 2324m 간에 각도가 예각인 경우
    frontUpAngle, frontSideAngle = frontShoulderPlaneAngle(
        mainShoulder=v11,
        otherShoulder=v12,
        v23=v23,
        v24=v24,
        shoulderPoint=shoulderPoint,
        headPoint=headPoint,
    )

    # 조건 체크
    sideOkay: bool = sideAngle < 120
    frontUpOkay: bool = frontUpAngle < 108
    frontSideOkay: bool = frontSideAngle < 110

    code: int = 0 if sideOkay and frontUpOkay and frontSideOkay else 1
    # if code !=0:
    #     st.text(f"{self.skeletonData.frameTime/6}.left -> sideAngle:{sideAngle}\nfrontUpOkay:{frontUpAngle}\nfrontSideOkay:{frontSideAngle}")
    # logger.info(f"  => sideAngle:{sideAngle},frontUpAngle:{frontUpAngle},frontSideAngle:{frontSideAngle} -> return code: {code}") #@ 테스트 로그
    return code


#@njit
def CheckRightArm(idx: int, array33: np.ndarray, v14: np.ndarray) -> int:
    """"""
    # 포인트 불러오기
    v11 = array33[11]
    v12 = array33[12]
    v16 = array33[16]

    v23 = array33[23]
    v24 = array33[24]
    m1112: np.ndarray = (v11 + v12) / 2
    m2324: np.ndarray = (v23 + v24) / 2
    bodyVector = m1112 - m2324

    # 어깨 근육 방향 벡터 추출 유효성 체크 : 팔을 핀 경우에는 어깨근육 벡터가 상당히 모호해짐
    armAngle: float = GetVectorAngle(v12 - v14, v16 - v14)

    if armAngle > 175:
        # print("펴져있어서 어깨근육으로 판별 불가하여 정상으로 반환")
        # logger.info(f"{idx}.CheckRightArm -> armAngle > 175 -> return 0") #@ 테스트 로그
        return 0

    # 어깨 근육 방향 벡터 추출
    shoulderVector: np.ndarray = GetNormalVector(v16 - v14, v12 - v14)
    shoulderPoint: np.ndarray = shoulderVector + v12

    # 머리위로 점
    headPoint = bodyVector + v12
    # logger.info(f"{idx}.CheckRightArm -> v12-v14,v16-v14 : {v12-v14,v16-v14},shoulderVector:{shoulderVector},shoulderPoint:{shoulderPoint},headPoint:{headPoint},bodyVector:{bodyVector}") #@ 테스트 로그

    # 측면 평면 : 머리 쪽 투영해서 예각인 경우에만 오케이
    sideAngle: float = SideShoulderPlaneAngle(
        mainShoulder=v12,
        otherShoulder=v11,
        shoulderPoint=shoulderPoint,
        headPoint=headPoint,
    )

    # 정면 평면 : 1112와 각도값이 예각이면서 1112m 과 2324m 간에 각도가 예각인 경우
    frontUpAngle, frontSideAngle = frontShoulderPlaneAngle(
        mainShoulder=v12,
        otherShoulder=v11,
        v23=v23,
        v24=v24,
        shoulderPoint=shoulderPoint,
        headPoint=headPoint,
    )

    # 조건 체크
    sideOkay: bool = sideAngle < 120
    frontUpOkay: bool = frontUpAngle < 108
    frontSideOkay: bool = frontSideAngle < 110

    code: int = 0 if sideOkay and frontUpOkay and frontSideOkay else -1
    # if code !=0:
    #     st.text(f"{self.skeletonData.frameTime/6}.right -> sideAngle:{sideAngle}\nfrontUpOkay:{frontUpAngle}\nfrontSideOkay:{frontSideAngle}")
    # logger.info(f"  => sideAngle:{sideAngle},frontUpAngle:{frontUpAngle},frontSideAngle:{frontSideAngle} -> return code: {code}") #@ 테스트 로그
    return code


#@njit
def OverCorrect(
    idx: int,
    array33: np.ndarray,
    array33_pre: np.ndarray,
    idxList: list[int],
    isLeft: bool,
):
    """내적 보정 알고리즘"""
    # 신규보정방법
    # 보정 종류 : 중점 보정(M(13)), 반전 보정(L(15) or M(13))
    # 1) 문제 발생시  앞, 뒤 여부 상관없이, 중점보정과 반전 보정 둘다 진행함
    # 2) 반전 보정시에는 M 중심으로 반전한 후에도 문제 발생하면, L중심 반전으로 전환
    # 3) 보정 대상자 지정
    # 4) 비교가 필요시 이전 분석 데이터(이전프레임)를 불러와서, 보정 대상자 핀포인트간의 거리를 측정함
    # 5) 가장 짧은 거리를 가진게 정답으로 판명함 (비교대상자인 이전값이 없으면 중심값을 우선으로 판정)

    # 중점 보정
    AbleMiddle: bool = True  # 중점 보정 유효성 체크

    correctMiddlePoint: np.ndarray = CorrectMiddle(array33[idxList[0]], array33[idxList[1]], array33[idxList[2]])

    if isLeft:
        # logger.info(f"{idx}.OverCorrect({idxList}) -> Left(M) -> {correctMiddlePoint}") #@ 테스트 로그
        ### 중심기반으로 보정했는데 다시 문제가 발생한 경우 -> 보정 대상 탈락
        # logger.info(f"{idx}.OverCorrect({idxList}) -> Left(M) -> {correctMiddlePoint}") #@ 테스트 로그
        if CheckLeftArm(idx, array33, correctMiddlePoint) != 0:
            # print(f"{self.skeletonData.frameTime/6}.왼팔: 중심보정시 문제 발생하여 중심 보정 제외")
            # logger.info(f"  => AbleMiddle=False") #@ 테스트 로그
            AbleMiddle = False

    else:
        # logger.info(f"{idx}.OverCorrect({idxList}) -> Right(M) -> {correctMiddlePoint}") #@ 테스트 로그
        ### M 중심기반으로 보정했는데 다시 문제가 발생한 경우 -> 보정 대상 탈락
        if CheckRightArm(idx, array33, correctMiddlePoint) != 0:
            # print(f"{self.skeletonData.frameTime/6}.오른팔: 중심보정시 문제 발생하여 중심 보정 제외")
            # logger.info(f"  => AbleMiddle=False") #@ 테스트 로그
            AbleMiddle = False

    # M 반전 보정
    ReverseTargetIndex: int = idxList[1]
    correctReversePoint: np.ndarray = CorrectReverse(array33[idxList[0]], array33[idxList[1]])
    if isLeft:
        # logger.info(f"{idx}.OverCorrect({idxList}) -> Left(R) -> {correctReversePoint}") #@ 테스트 로그
        ### M 중심으로 보정했는데 다시 문제가 발생한 경우 -> L 보정으로 전환
        if CheckLeftArm(idx, array33, v13=correctReversePoint) != 0:
            # print(f"{self.skeletonData.frameTime/6}.왼팔: L보정으로 전환")
            ReverseTargetIndex = idxList[2]
            correctReversePoint = CorrectReverse(array33[idxList[1]], array33[idxList[2]])
            # logger.info(f"  => ReverseTargetIndex:{ReverseTargetIndex}, correctReversePoint:{correctReversePoint}") #@ 테스트 로그

    else:
        # logger.info(f"{idx}.OverCorrect({idxList}) -> Right(R) -> {correctReversePoint}") #@ 테스트 로그
        ### M 중심으로 보정했는데 다시 문제가 발생한 경우 -> L 보정으로 전환
        if CheckRightArm(idx, array33, v14=correctReversePoint) != 0:
            # print(f"{self.skeletonData.frameTime/6}.오른팔: L보정으로 전환")
            ReverseTargetIndex = idxList[2]
            correctReversePoint = CorrectReverse(array33[idxList[1]], array33[idxList[2]])
            # logger.info(f"  => ReverseTargetIndex:{ReverseTargetIndex}, correctReversePoint:{correctReversePoint}") #@ 테스트 로그≈

    if AbleMiddle:
        if array33_pre is None:
            # 만약에 이전값이 없다면?
            array33[idxList[1]] = correctMiddlePoint
            return

        # 둘이 비교해야하는 경우
        d1: float = Distance(array33_pre[idxList[1]], correctMiddlePoint)
        d2: float = Distance(array33_pre[ReverseTargetIndex], correctReversePoint)

        # print(f"d1: {d1} , d2 : {d2}")

        if d1 > d2:
            # d2가 작기에, 반전 보정 채택
            # print(f"최종 : 반전 보정 채택 {ReverseTargetIndex}")
            array33[ReverseTargetIndex] = correctReversePoint
            # logger.info(f"      if AbleMiddle ->  d1:{d1},d2:{d2} ->{ReverseTargetIndex} -> {correctReversePoint}") #@ 테스트 로그
        else:
            # 중점 보정 채택
            # print("최종 : 중점 보정 채택")
            array33[idxList[1]] = correctMiddlePoint
            # logger.info(f"      if AbleMiddle ->  d1:{d1},d2:{d2} ->{idxList[1]} -> {correctMiddlePoint}") #@ 테스트 로그

        return

    # 비교안하고 반전 보정 결과를 받아오는 경우 (플랫폼별 변환 필요)
    # print(f"최종 : 반전 보정 채택 {ReverseTargetIndex}")
    array33[ReverseTargetIndex] = correctReversePoint


############################################################################################################
# 보정 방법
############################################################################################################
#@njit
def CorrectMiddle(start: np.ndarray, middle: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Z값 중점 보정"""
    middleZ: float = (start[2] + end[2]) / 2
    return np.array([middle[0], middle[1], middleZ])  # PC는 해상도 16:9 이슈로 x값에 1.8을 곱함


#@njit
def CorrectReverse(root: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Z값 반전 보정"""
    addValue: float = root[2] - target[2]
    return np.array([target[0], target[1], (root[2] + addValue)])  # PC는 해상도 16:9 이슈로 x값에 1.8을 곱함


############################################################################################################
# 기타 유틸함수
############################################################################################################


#@njit
def Distance(t1: np.ndarray, t2: np.ndarray) -> float:
    """두 점간에 거리측정"""
    return np.linalg.norm(t2 - t1)
    # return Point([t1[0], t1[1],t1[2]]).distance_point([t2[0], t2[1],t2[2]]) # 변환하기 전 처리값


#@njit
def RangeCheck(a: float, b: float) -> bool:
    """3도 오차 체크"""
    return abs(b - a) > 3


#@njit
def RangeFrontCheck(arm: np.ndarray, bodyFront: np.ndarray) -> bool:
    """앞쪽 판정 체크"""
    angle: float = GetVectorAngle(arm, bodyFront)
    if 87 < angle and angle < 93:
        return False
    else:
        return angle > 90  # True -> 뒤로 판정


# np array 벡터 노말라이징
#@njit
def normalize(vector):
    norm = np.linalg.norm(vector)
    if norm == 0.0:
        return vector

    return vector / norm


#@njit
def SideShoulderPlaneAngle(
    mainShoulder: np.ndarray,
    otherShoulder: np.ndarray,
    shoulderPoint: np.ndarray,
    headPoint: np.ndarray,
) -> float:
    """어깨 옆면 기반으로 어깨근육 방향 각도 추출"""

    # 측면 평면 : 머리 쪽 투영해서 예각인 경우에만 오케이
    ShoulderLine: np.ndarray = otherShoulder - mainShoulder

    project_side_shoulder_point = PlaneProjectionNormal(mainShoulder, ShoulderLine, shoulderPoint)
    project_side_head_point = PlaneProjectionNormal(mainShoulder, ShoulderLine, headPoint)

    project_side_target_vector: np.ndarray = project_side_shoulder_point - mainShoulder
    project_side_base_vector: np.ndarray = project_side_head_point - mainShoulder

    # 주요 값 추출 : 사이드 평면상에서의 각도
    sidePlaneAngle: float = GetVectorAngle(project_side_target_vector, project_side_base_vector)
    return sidePlaneAngle


#@njit
def frontShoulderPlaneAngle(
    mainShoulder: np.ndarray,
    otherShoulder: np.ndarray,
    v23: np.ndarray,
    v24: np.ndarray,
    shoulderPoint: np.ndarray,
    headPoint: np.ndarray,
):
    """"""
    m2324: np.ndarray = (v23 + v24) / 2

    # 정면 평면 : 1112와 각도값이 예각이면서 1112m 과 2324m 간에 각도가 예각인 경우

    project_front_shoulder_point = PlaneProjection(mainShoulder, otherShoulder, m2324, shoulderPoint)
    project_front_head_point = PlaneProjection(mainShoulder, otherShoulder, m2324, headPoint)

    project_front_target_vector: np.ndarray = project_front_shoulder_point - mainShoulder
    project_front_base_vector: np.ndarray = project_front_head_point - mainShoulder

    # 머리 기준으로 각도 : 예각이여야함
    frontUpAngle: float = GetVectorAngle(project_front_target_vector, project_front_base_vector)

    # 머리 기준으로 각도 : 예각이여야함
    frontSideAngle: float = GetVectorAngle(project_front_target_vector, mainShoulder - otherShoulder)

    return frontUpAngle, frontSideAngle


#########################################################################################################
# 벡터 투영 관련 연산
#########################################################################################################


#@njit
def PlaneProjection(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: np.ndarray) -> np.ndarray:
    # 평면 벡터 투영 : 3점을 기반으로 평면 생성하여, t를 투영
    plane = (p1, p2, p3)

    a = plane[1] - plane[0]
    b = plane[2] - plane[0]

    v = np.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    )
    d = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    u = v / d

    d = np.sum((t - plane[0]) * u)
    pp = t - (u * d)
    return pp


#@njit
def PlaneProjectionNormal(a: np.ndarray, n: np.ndarray, t: np.ndarray) -> np.ndarray:
    # 평면 벡터 투영(노멀값기반)) : 1점과 노멀벡터 기반으로 평면 생성하여, t를 투영
    d1 = math.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
    n = n / d1
    d = np.sum((t - a) * n)
    pp = t - (n * d)

    return pp
