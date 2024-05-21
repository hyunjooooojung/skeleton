import random

# 벡터 데이터 추출
def GetVectorData(angle):
    return angle[1]+angle[3]+angle[5]
    

# 1번항목
def get_both_vectors(all_landmarks):
    
    def create_vector_from_landmarks(all_landmarks, index1, index2):
        # 첫 번째 랜드마크의 x, y, z 좌표
        x1, y1, z1 = all_landmarks[index1]['skeletons'][0]['x'], all_landmarks[index1]['skeletons'][0]['y'], all_landmarks[index1]['skeletons'][0]['z']
        
        # 두 번째 랜드마크의 x, y, z 좌표
        x2, y2, z2 = all_landmarks[index2]['skeletons'][0]['x'], all_landmarks[index2]['skeletons'][0]['y'], all_landmarks[index2]['skeletons'][0]['z']
        
        # 두 랜드마크 사이의 벡터 계산
        vector = (x2 - x1, y2 - y1, z2 - z1)
        return vector

    # all_landmarks 데이터에서 벡터 생성
    vector_between_landmarks1 = create_vector_from_landmarks(all_landmarks, 13, 15)
    vector_between_landmarks2 = create_vector_from_landmarks(all_landmarks, 14, 16)
    print("계산된 벡터:", vector_between_landmarks1,",",vector_between_landmarks2)
    
    return vector_between_landmarks1,vector_between_landmarks2

def get_z_vector(num:str):
    
    if num == "1":
        rr = [0,0,1,0,0,1]
    elif num == "2":
        rr = [0,0,1,0,0,1]
    elif num == "3":
        rr = [0,0,-1,0,0,-1]
    elif num == "4":
        rr = [0,0,-1,0,0,-1]
    elif num == "5":
        rr = [0,0,1,0,0,1]
    elif num == "6":
        rr = [0,0,1,0,0,-1]
    elif num == "7":
        rr = [0,0,1,0,0,1]
    elif num == "8":
        rr = [0,0,1,0,0,1]
    elif num == "9":
        rr = [0,0,1,0,0,1]
    elif num == "10":
        rr = [0,0,1,0,0,1]
    elif num == "11":
        rr = [0,0,1,0,0,-1]
    elif num == "12":
        rr = [0,0,-1,0,0,1]
    elif num == "13":
        rr = [0,0,1,0,0,1]
    elif num == "14":
        rr = [0,0,1,0,0,-1]
    elif num == "15":
        rr = [0,0,1,0,0,-1]
    elif num == "16":
        rr = [0,0,1,0,0,1]
    elif num == "17":
        rr = [0,0,1,0,0,-1]
    elif num == "18":
        rr = [0,0,-1,0,0,-1]
    elif num == "19":
        rr = [0,0,-1,0,0,1]
    elif num == "20":
        rr = [0,0,1,0,0,1]
    else :
        rr = [0,0,1,0,0,1]
    
    part1 = rr[:3]
    part2 = rr[3:]
    
    return part1,part2


def getRatio(num,l1,l2,l3,l4):
    if num == "1":
        rr = [ 0.154,0.159,0.228,0.226]
    elif num == "2":
        rr = [0.134,0.132,0.200,0.195]
    elif num == "3":
        rr = [0.10,0.131,0.192,0.190]
    elif num == "4":
        rr = [0.142,	0.139,	0.195,	0.195]
    elif num == "5":
        rr = [0.124,	0.143,	0.213,	0.214]
    elif num == "6":
        rr = [0.138,	0.141,	0.223,	0.226]
    elif num == "7":
        rr = [0.132,0.136,0.209,0.21]
    elif num == "8":
        rr = [0.138,	0.134	,0.195,	0.197]
    elif num == "9":
        rr = [0.155,	0.158,	0.221,	0.218]
    elif num == "10":
        rr = [0.128,	0.144,	0.202,	0.196]
    else :
        rr=[l1,l2,l3,l4]
        return rr
        
    return [r+0.0025+(0.0025) * random.random()  for r in rr]


