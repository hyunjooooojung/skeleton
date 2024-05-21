from __future__ import annotations
import random
import numpy as np
import pandas as pd
from lib.model import  Skeleton, SkeletonData
from skspatial.objects import Point

class RatioModel:
    ratio:float
    n1:int
    n2:int
    length1:int
    '''n1-n2에 상위 위치의 길이데이터의 인덱스(상단)'''
    length2:int
    '''n1-n2에 해당하는 길이데이터의 인덱스(하단)'''


    def __init__(self,ratio:float,n1:int,n2:int,length1:int,length2:int):
        self.ratio=ratio +random.uniform(-0.005, 0.005)
        self.n1=n1
        self.n2=n2
        self.length1=length1
        self.length2=length2

    def Show(self):
        return f"RatioModel: {self.n1},{self.n1}"
    
    def get(self):
        return self.ratio


def fixRatioSystem(skeletonData:SkeletonData,array:list[float]):
    '''전체비율고정'''
    rm1=RatioModel(array[0],n1=13,n2=15,length1=1,length2=2)
    rm2=RatioModel(array[1],n1=14,n2=16,length1=3,length2=4)
    rm3=RatioModel(array[2],n1=25,n2=27,length1=5,length2=6)
    rm4=RatioModel(array[3],n1=26,n2=28,length1=7,length2=8)

    rms:list[RatioModel]=[rm1,rm2,rm3,rm4]

    # 비율 변경
    fixRatio(skeletonData,rms)   
          

    return skeletonData,rms
    # 변경된 부분은 Cz에 반영
    
    
def fixRatio(skeletonData:SkeletonData,rms:list[RatioModel]):
    '''비율 고정'''

    # 해당 프레인의 길이값 반환 
    bodylengths:list[float]=getBodyLengths(skeletonData)

    # 스켈레톤 데이터 
    skeletons=skeletonData.skeletons

    ratioModel:RatioModel
    # 바꿔야하는 값 기반으로 불러오기
    for ratioModel in rms:  
        
        # 변경에 적용되는 제원
        n1:int=ratioModel.n1
        n2:int=ratioModel.n2
        length1:int=ratioModel.length1
        length2:int=ratioModel.length2
        ratio:float=ratioModel.ratio

        # 대상 길이 추출
        l1_length:float=bodylengths[length1] # 기준이 되는 상단의 길이 
        l2_length:float=bodylengths[length2] # 원래길이
        lengthPlane:float=getLengthByPlane(skeletons=skeletons,n1=n1,n2=n2)

        # 바꿔야하는 길이
        l2_target=l1_length*ratio # 바뀌는 길이

        # 바뀌는값
        change_z_value:float=0

        if l2_length<l2_target:
            # Pre<Pro

            value_pow2=l2_target*l2_target-lengthPlane*lengthPlane

            if value_pow2 <0:
                # 평면값이 타겟값보다 더 큰경우(0.9배에 의해 발생) : 상위값과 동일한 Z값으로 적용
                change_z_value=skeletons[n1].z        
            else :
                # 피타고라스 및 크기에 따라 적용
                value=value_pow2**0.5
                isPlus=skeletons[n2].z>skeletons[n1].z
                change_z_value=skeletons[n1].z+(value if isPlus else -value)

        elif l2_length>l2_target:
            # Pre>Pro

            value_pow2=l2_target*l2_target-lengthPlane*lengthPlane

            
            if value_pow2 <0:
                # 평면값이 타겟값보다 더 큰경우(0.9배에 의해 발생) : 상위값과 동일한 Z값으로 적용
                change_z_value=skeletons[n1].z        
            else :
                # 피타고라스 및 크기에 따라 적용
                value=value_pow2**0.5
                isPlus=skeletons[n2].z>skeletons[n1].z
                change_z_value=skeletons[n1].z+(value if isPlus else -value)

        # 반영하기
        skeletons[n2].cz=change_z_value

def getLengthByPlane(skeletons:list[Skeleton],n1:int,n2:int):
    '''평면기준 길이 값'''
    p1=skeletons[n1].GetTupleOriginalXY()
    p2=skeletons[n2].GetTupleOriginalXY()

    distance:float=Point(p1).distance_point(Point(p2))

    return distance


 


################################################################################################
################################################################################################
################################################################################################    

def getPandasBodyRatio(skeletonDatas:list[SkeletonData]):
    '''판다스 데이터로 변환 : 비율'''
 

    # 바디 데이터 배열 반환
    bodyratiosArray:list[list[float]]=getAllBodyRatios(skeletonDatas)    

    # 판다스 데이터로 변환
    df=pd.DataFrame(bodyratiosArray)

    return df


def getAllBodyRatios(skeletonDatas:list[SkeletonData]):
    '''전체 비율 추출'''

    # 전체 바디 렝스 데이트 추출
    bodylengthsArray:list[list[float]]=[getBodyLengths(skeletonData) for skeletonData in skeletonDatas]

    # 전체 바디 렝스 기반으로 비율 추출
    bodyratiosArray:list[list[float]]=[getBodyRatio(bodylengths) for bodylengths in bodylengthsArray]
    
    return bodyratiosArray


def getBodyRatio(lengths:list[float]):
    '''비율 추출'''

    # 왼팔
    arm_l=lengths[1]/lengths[2]
    
    
    # 오른팔
    arm_r=lengths[3]/lengths[4]

    # 왼쪽다리
    leg_l=lengths[5]/lengths[6]    
    
    # 오른쪽다리
    leg_r=lengths[7]/lengths[8]
    

    array:list[float]=[arm_l,arm_r,leg_l,leg_r]

    return array

################################################################################################
################################################################################################
################################################################################################

def getPandasBodyLengths(skeletonDatas:list[SkeletonData]):
    '''판다스 데이터로 변환 : 길이'''
    
    # 바디 데이터 배열 반환
    bodylengthsArray:list[list[float]]=getAllBodyLengths(skeletonDatas)

    # 판다스 데이터로 변환
    df=pd.DataFrame(bodylengthsArray)

    return df

def getAllBodyLengths(skeletonDatas:list[SkeletonData]):
    '''전체 길이 추출'''

    # 전체 바디 렝스 데이트 추출
    bodylengthsArray:list[list[float]]=[getBodyLengths(skeletonData) for skeletonData in skeletonDatas]
    
    return bodylengthsArray




def getBodyLengths(skeletonData:SkeletonData)->list[float]:
    '''길이 추출 : 8개 길이 추출'''

    # 각 관절별 길이 만드는 곳 : 여기 @@@@ 그래프 그릴수있도록 만들고, 관절 번호도 1,2가아닌 다른것들 넣어주기

    # 골반 길이 : 기준길이
    pelvis:float=__getDistance(skeletonData,n1=23,n2=24)

    # 왼팔
    arm_l1:float=__getDistance(skeletonData,n1=11,n2=13)
    arm_l2:float=__getDistance(skeletonData,n1=13,n2=15)

    # 오른팔
    arm_r1:float=__getDistance(skeletonData,n1=12,n2=14)
    arm_r2:float=__getDistance(skeletonData,n1=14,n2=16)

    # 왼쪽다리
    leg_l1:float=__getDistance(skeletonData,n1=23,n2=25)
    leg_l2:float=__getDistance(skeletonData,n1=25,n2=27)
    
    # 오른쪽다리
    leg_r1:float=__getDistance(skeletonData,n1=24,n2=26)
    leg_r2:float=__getDistance(skeletonData,n1=26,n2=28)

    # Np 배열로 저장
    array:list[float]=[pelvis,arm_l1,arm_l2,arm_r1,arm_r2,leg_l1,leg_l2,leg_r1,leg_r2]
    return array






############################################################################################################
############################################################################################################
############################################################################################################

def __getDistance(skeletonData:SkeletonData,n1:int,n2:int):
    '''길이 추출'''
    
    # 예시안        
    skeleton1:Skeleton=skeletonData.skeletons[n1]
    skeleton2:Skeleton=skeletonData.skeletons[n2]


    # 추출 포인트
    point1:Point=__getPoint(skeleton1)
    point2:Point=__getPoint(skeleton2)

    distnace:float=point1.distance_point(point2) 
    
    # 계산 수식 에러 방지
    if distnace==None:
        distnace=0

    return distnace

def __getPoint(skeleton:Skeleton)->Point:
    '''3D 포인트 추출'''
    return Point([skeleton.x,skeleton.y,skeleton.z])    


 