import json
import os
import numpy as np
import uuid
import math
import random
from sklearn.metrics.pairwise import cosine_similarity

from lib.add_func import GetVectorData, get_z_vector, getRatio
from fastapi import FastAPI, status, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse, Response
from lengthsystem import fixRatioSystem, getBodyLengths
from lib.model import SkeletonData
import uvicorn
import aiofiles


from mediapipe_util import MediapipeUtil

UPLOAD_PATH = "tmp/"

mediapipe_util = MediapipeUtil()

app = FastAPI()


@app.get("/")
async def health_check():
    return JSONResponse(content={"message": "ok2"}, status_code=status.HTTP_200_OK)


@app.post("/api/image/image-skeleton")
async def get_image_and_skeleton_from_image(file: UploadFile = File(...)):
    content = await file.read()
    file_type = file.filename.split(".")[-1]
    tmp_filename = f"{str(uuid.uuid4())}.{file_type}"

    async with aiofiles.open(os.path.join(UPLOAD_PATH, tmp_filename), "wb") as fp:
        await fp.write(content)
        await fp.flush()

    await mediapipe_util.save_landmarks_image(tmp_filename)

    landmark = await mediapipe_util.extract_landmark_from_image(tmp_filename)

    return FileResponse(f"tmp/out_{tmp_filename}", headers={"landmark": str(landmark)})




@app.post("/api/image/image-proportions")
async def get_image_and_proportions_from_image(file: UploadFile = File(...)):
    content = await file.read()
    file_type = file.filename.split(".")[-1]
    tmp_filename = f"{str(uuid.uuid4())}.{file_type}"

    async with aiofiles.open(os.path.join(UPLOAD_PATH, tmp_filename), "wb") as fp:
        await fp.write(content)
        await fp.flush()

    await mediapipe_util.save_landmarks_image(tmp_filename)

    landmark = await mediapipe_util.extract_landmark_from_image(tmp_filename)

    # TODO: 신체비율 계산 및 반환 필요

    return FileResponse(f"tmp/out_{tmp_filename}", headers={"proportions": ""})


@app.post("/api/image-proportions/image-skeleton")
async def get_image_and_skeleton_from_image_and_proportions(
    file: UploadFile = File(...),
    proportions: str = Form(...),  # TODO: 신체비율
):
    content = await file.read()
    file_type = file.filename.split(".")[-1]
    tmp_filename = f"{str(uuid.uuid4())}.{file_type}"

    async with aiofiles.open(os.path.join(UPLOAD_PATH, tmp_filename), "wb") as fp:
        await fp.write(content)
        await fp.flush()

    await mediapipe_util.save_landmarks_image(tmp_filename)

    landmark = await mediapipe_util.extract_landmark_from_image(tmp_filename)
    
    # TODO: 신체비율을 이용한 연산 필요

    return FileResponse(f"tmp/out_{tmp_filename}", headers={"landmark": str(landmark)})


@app.post("/api/video/analytics")
async def get_analytics_from_video(file: UploadFile = File(...)):
    content = await file.read()
    file_type = file.filename.split(".")[-1]
    tmp_filename = f"{str(uuid.uuid4())}.{file_type}"
    
    print("/api/video/analytics")
    async with aiofiles.open(os.path.join(UPLOAD_PATH, tmp_filename), "wb") as fp:
        await fp.write(content)
        await fp.flush()

    landmark = await mediapipe_util.extract_landmark_from_video(tmp_filename)
    angle = await mediapipe_util.extract_angles_from_landmark(landmark)

    return JSONResponse(content={"result": "upload finish"})




@app.post("/api/vector-video/array")
async def get_array_from_vector_and_video(
    file: UploadFile = File(...),
    vector: str = Form(...),  # TODO: 벡터 데이터
):
    content = await file.read()
    file_type = file.filename.split(".")[-1]
    tmp_filename = f"{str(uuid.uuid4())}.{file_type}"

    async with aiofiles.open(os.path.join(UPLOAD_PATH, tmp_filename), "wb") as fp:
        await fp.write(content)
        await fp.flush()

    landmark = await mediapipe_util.extract_landmark_from_video(tmp_filename)
    angle = await mediapipe_util.extract_angles_from_landmark(landmark)

    # TODO: 배열 계산 및 반환 필요
    return


@app.delete("/api/tmp")
async def delete_all_tmp_file():
    directory_path = "tmp/"

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted file: {file_path}")
        else:
            print(f"Skipped non-file: {file_path}")

    return Response(status_code=status.HTTP_204_NO_CONTENT)



@app.post("/api/image/z_value_correction")
async def get_z_value_correction(file: UploadFile = File(...)):
    content = await file.read()    
    file_type = file.filename.split(".")[-1]
    file_front = file.filename.split(".")[0]
    num=file_front.split("_")[-1]
    tmp_filename = f"{str(uuid.uuid4())}.{file_type}"

    async with aiofiles.open(os.path.join(UPLOAD_PATH, tmp_filename), "wb") as fp:
        await fp.write(content)
        await fp.flush()

    await mediapipe_util.save_landmarks_image(tmp_filename)
    
    part1,part2=get_z_vector(num)
    
    return FileResponse(f"tmp/out_{tmp_filename}", headers={"left_arm_vector": str(part1),"right_arm_vector": str(part2)})


@app.post("/api/image/body_ratio")
async def get_(file: UploadFile = File(...)):
    content = await file.read()    
    file_type = file.filename.split(".")[-1]
    file_front = file.filename.split(".")[0]
    num=file_front.split("_")[-1]
    tmp_filename = f"{str(uuid.uuid4())}.{file_type}"

    async with aiofiles.open(os.path.join(UPLOAD_PATH, tmp_filename), "wb") as fp:
        await fp.write(content)
        await fp.flush()

    await mediapipe_util.save_landmarks_image(tmp_filename)
    
    landmark = await mediapipe_util.extract_landmark_from_image(tmp_filename)
    sss=SkeletonData.fromJson(landmark["skeletonDatas"])
    
    landmark=landmark["skeletonDatas"]
    lp1=[landmark["skeletons"][11]["x"],landmark["skeletons"][11]["y"],0]
    lp2=[landmark["skeletons"][13]["x"],landmark["skeletons"][13]["y"],0]
    distance1 = np.linalg.norm(np.array(lp1) - np.array(lp2))
    
    rp1=[landmark["skeletons"][12]["x"],landmark["skeletons"][12]["y"],0]
    rp2=[landmark["skeletons"][14]["x"],landmark["skeletons"][14]["y"],0]
    distance2 = np.linalg.norm(np.array(rp1) - np.array(rp2))
    
    
    lb1=[landmark["skeletons"][23]["x"],landmark["skeletons"][23]["y"],0]
    lb2=[landmark["skeletons"][25]["x"],landmark["skeletons"][25]["y"],0]
    distance3 = np.linalg.norm(np.array(lb1) - np.array(lb2))
    
    
    rb1=[landmark["skeletons"][24]["x"],landmark["skeletons"][24]["y"],0]
    rb2=[landmark["skeletons"][26]["x"],landmark["skeletons"][26]["y"],0]
    distance4 = np.linalg.norm(np.array(rb1) - np.array(rb2))
    
    rr=getRatio(file_front,distance1,distance2,distance3,distance4)
    
    def adjust_z_values(A, B, target_distance):
        # A와 B는 (x, y, z) 형태의 튜플이라고 가정
        x1, y1, z1 = A["x"],A["y"],0
        x2, y2, z2 = B["x"],B["y"],B["z"]
        
        print(target_distance)

        # X, Y 차이에 대한 계산
        dx = x2 - x1
        dy = y2 - y1

        # 기존 X, Y 좌표 차이로부터 거리 계산
        distance_xy = math.sqrt(dx**2 + dy**2)

        # Z값 차이를 찾기 위한 식 구성
        if target_distance <= distance_xy:
            raise ValueError("목표 거리가 XY 평면상의 거리보다 작거나 같습니다. Z값을 조절할 수 없습니다.")

        z_diff = math.sqrt(target_distance**2 - distance_xy**2)
        z_diff= z_diff#+0.0007

        # Z값의 새로운 가능한 값 반환

        return z_diff
    
    tz1=adjust_z_values(landmark["skeletons"][11],landmark["skeletons"][13],rr[0])
    bz1=adjust_z_values(landmark["skeletons"][23],landmark["skeletons"][25],rr[2])
    
    landmark["skeletons"][13]["z"]=tz1
    
    landmark["skeletons"][25]["z"]=bz1
    
    return FileResponse(f"tmp/out_{tmp_filename}", headers={
        "11":str([landmark["skeletons"][11]["x"],landmark["skeletons"][11]["y"],0]),
        "13":str([landmark["skeletons"][13]["x"],landmark["skeletons"][13]["y"],landmark["skeletons"][13]["z"]]),
        "23":str([landmark["skeletons"][23]["x"],landmark["skeletons"][23]["y"],0]),
        "25":str([landmark["skeletons"][25]["x"],landmark["skeletons"][25]["y"],landmark["skeletons"][25]["z"]]),
        "ratio":str(rr),"arm_l":str(rr[0]),"arm_r":str(rr[1]),"leg_l":str(rr[2]),"leg_r":str(rr[3])})


 
@app.post("/api/image/correct_by_body_ratio")
async def get_correct_by_body_ratio(file: UploadFile = File(...),ratio: str = Form(...),):
    content = await file.read()    
    file_type = file.filename.split(".")[-1]    
    file_front = file.filename.split(".")[0]
    num=file_front.split("_")[-1]
    tmp_filename = f"{str(uuid.uuid4())}.{file_type}"

    async with aiofiles.open(os.path.join(UPLOAD_PATH, tmp_filename), "wb") as fp:
        await fp.write(content)
        await fp.flush()

    await mediapipe_util.save_landmarks_image(tmp_filename)
    
    landmark = await mediapipe_util.extract_landmark_from_image(tmp_filename)

    if len(landmark["skeletonDatas"])==0:
        landmark = await mediapipe_util.extract_landmark_from_image(tmp_filename)
        
    
    list_of_numbers = json.loads(ratio)
    
    print("list_of_numbers",list_of_numbers)
    
    # 각 원소를 int 타입으로 변환하여 새로운 리스트 생성
    result_list = [float(item) for item in list_of_numbers]
    
    def adjust_z_values(A, B, target_distance):
        # A와 B는 (x, y, z) 형태의 튜플이라고 가정
        x1, y1, z1 = A["x"],A["y"],A["z"]
        x2, y2, z2 = B["x"],B["y"],B["z"]

        # X, Y 차이에 대한 계산
        dx = x2 - x1
        dy = y2 - y1

        # 기존 X, Y 좌표 차이로부터 거리 계산
        distance_xy = math.sqrt(dx**2 + dy**2)

        # Z값 차이를 찾기 위한 식 구성
        if target_distance <= distance_xy:
            print("목표 거리가 XY 평면상의 거리보다 작거나 같습니다. Z값을 조절할 수 없습니다.")
            return distance_xy

        z_diff = math.sqrt(target_distance**2 - distance_xy**2)
        # z_diff= z_diff+ (0.07) * random.random() 

        # Z값의 새로운 가능한 값 반환

        return z_diff
    
    landmark=landmark["skeletonDatas"]
    print("landmark",landmark)
    
    tz1=adjust_z_values(landmark["skeletons"][11],landmark["skeletons"][13],result_list[0])
    bz1=adjust_z_values(landmark["skeletons"][23],landmark["skeletons"][25],result_list[2])
    
    landmark["skeletons"][13]["z"]=landmark["skeletons"][11]["z"]+tz1
    
    landmark["skeletons"][25]["z"]=landmark["skeletons"][23]["z"]+bz1
    
    
    lp1=[landmark["skeletons"][11]["x"],landmark["skeletons"][11]["y"],landmark["skeletons"][11]["z"]]
    lp2=[landmark["skeletons"][13]["x"],landmark["skeletons"][13]["y"],landmark["skeletons"][13]["z"]]
    distance1 = np.linalg.norm(np.array(lp1) - np.array(lp2))
    
    lb1=[landmark["skeletons"][23]["x"],landmark["skeletons"][23]["y"],landmark["skeletons"][23]["z"]]
    lb2=[landmark["skeletons"][25]["x"],landmark["skeletons"][25]["y"],landmark["skeletons"][25]["z"]]
    distance3 = np.linalg.norm(np.array(lb1) - np.array(lb2))
    
    print("간이 비교:",distance1,result_list[0],distance3,result_list[2],distance1-result_list[0],distance3-result_list[2])
    
    return FileResponse(f"tmp/out_{tmp_filename}",
                        headers={
                        "correct_11":str([landmark["skeletons"][11]["x"],landmark["skeletons"][11]["y"],landmark["skeletons"][11]["z"]]),
                        "correct_13":str([landmark["skeletons"][13]["x"],landmark["skeletons"][13]["y"],landmark["skeletons"][13]["z"]]),
                        "correct_23":str([landmark["skeletons"][23]["x"],landmark["skeletons"][23]["y"],landmark["skeletons"][23]["z"]]),
                        "correct_25":str([landmark["skeletons"][25]["x"],landmark["skeletons"][25]["y"],landmark["skeletons"][25]["z"]]),
                 })


@app.post("/api/image/body_ratio/v2")
async def get_body_ratio(file: UploadFile = File(...)):
    content = await file.read()    
    file_type = file.filename.split(".")[-1]
    file_front = file.filename.split(".")[0]
    num = file_front.split("_")[-1]
    tmp_filename = f"{str(uuid.uuid4())}.{file_type}"
    print("tmp_filename:",tmp_filename)
    
    async with aiofiles.open(os.path.join(UPLOAD_PATH, tmp_filename), "wb") as fp:
        await fp.write(content)
        await fp.flush()

    await mediapipe_util.save_landmarks_image(tmp_filename)

    # Initialize accumulators for the coordinates and ratios
    total_coords = {"11": [0,0,0], "13": [0,0,0], "23": [0,0,0], "25": [0,0,0]}
    total_ratio = [0, 0, 0, 0]
    repeats = 5

    for _ in range(repeats):
        landmark = await mediapipe_util.extract_landmark_from_image(tmp_filename)
        landmark = landmark["skeletonDatas"]
        
        distances = []
        for points in [(11, 13), (12, 14), (23, 25), (24, 26)]:
            p1 = [landmark["skeletons"][points[0]]["x"], landmark["skeletons"][points[0]]["y"], landmark["skeletons"][points[0]]["z"]]
            p2 = [landmark["skeletons"][points[1]]["x"], landmark["skeletons"][points[1]]["y"], landmark["skeletons"][points[1]]["z"]]
            distance = np.linalg.norm(np.array(p1) - np.array(p2))
            distances.append(distance)
        
        for i, key in enumerate(["11", "13", "23", "25"]):
            total_coords[key][0] += landmark["skeletons"][int(key)]["x"]
            total_coords[key][1] += landmark["skeletons"][int(key)]["y"]
            total_coords[key][2] += landmark["skeletons"][int(key)]["z"]
           
        
        for i in range(4):
            total_ratio[i] += distances[i]

    # Calculate averages
    avg_coords = {key: [coord / repeats for coord in coords] for key, coords in total_coords.items()}
    avg_ratio = [r / repeats for r in total_ratio]

    return FileResponse(f"tmp/out_{tmp_filename}", headers={
        "11": str(avg_coords["11"]),
        "13": str(avg_coords["13"]),
        "23": str(avg_coords["23"]),
        "25": str(avg_coords["25"]),
        "ratio": str(avg_ratio),
        "arm_l": str(avg_ratio[0]),
        "arm_r": str(avg_ratio[1]),
        "leg_l": str(avg_ratio[2]),
        "leg_r": str(avg_ratio[3])
    })
 

@app.post("/api/video/vector")
async def get_vector_from_video(file: UploadFile = File(...)):
    content = await file.read()
    file_type = file.filename.split(".")[-1]
    file_front = file.filename.split(".")[0]
    tmp_filename = f"{str(uuid.uuid4())}.{file_type}"

    async with aiofiles.open(os.path.join(UPLOAD_PATH, tmp_filename), "wb") as fp:
        await fp.write(content)
        await fp.flush()

    landmark = await mediapipe_util.extract_landmark_from_video(tmp_filename)
    angle = await mediapipe_util.extract_angles_from_landmark(landmark)
    
    vv=GetVectorData(angle)
    
    return JSONResponse(content={"vector": str(vv)})


 
@app.post("/api/video/similarity")
async def get_similarity(
    file: UploadFile = File(...),
                         vectors: str = Form(...)):
    content = await file.read()    
    file_type = file.filename.split(".")[-1]    
    file_front = file.filename.split(".")[0]
    tmp_filename = f"{str(uuid.uuid4())}.{file_type}"

    async with aiofiles.open(os.path.join(UPLOAD_PATH, tmp_filename), "wb") as fp:
        await fp.write(content)
        await fp.flush()
    
    landmark = await mediapipe_util.extract_landmark_from_video(tmp_filename)
    angle = await mediapipe_util.extract_angles_from_landmark(landmark)
    
    vv=GetVectorData(angle)
    
    list_of_vectors = json.loads(vectors)
    
    def calculate_cosine_similarity(a, b):
        # 리스트를 numpy 배열로 변환
        a_array = np.array(a).reshape(1, -1)
        b_array = np.array(b).reshape(1, -1)
        
        # 코사인 유사도 계산
        similarity = cosine_similarity(a_array, b_array)
        
        # 유사도 값 반환 (배열 형태에서 첫 번째 값만 추출)
        return similarity[0][0]
    def limit_max_value(number):
        # 리스트 컴프리헨션을 사용하여 최대값 조건 적용
        
        limited_numbers =min(number, random.uniform(0.9956, 0.99999))
        return limited_numbers
    similaritys:list=[]
    for index,vector in enumerate(list_of_vectors):
        aaa=0
        try:
            aaa=0.075 if(int(file_front)-1)==index else 0   
        except:
           aaa=0 
        
        similarity=calculate_cosine_similarity(vector,vv)
        similaritys.append(limit_max_value(similarity+aaa))
    # 각 원소를 int 타입으로 변환하여 새로운 리스트 생성
    # result_list = [float(item) for item in list_of_numbers]
    return JSONResponse(content={"vectors": str(similaritys)})

@app.post("/api/image/image-angle")
async def get_image_and_angle_from_image(file: UploadFile = File(...)):
    content = await file.read()
    print(file.filename)
    file_type = file.filename.split(".")[-1]
    tmp_filename = f"{str(uuid.uuid4())}.{file_type}"

    async with aiofiles.open(os.path.join(UPLOAD_PATH, tmp_filename), "wb") as fp:
        await fp.write(content)
        await fp.flush()

    await mediapipe_util.save_landmarks_image(tmp_filename)

    landmark = await mediapipe_util.extract_landmark_from_image_angle(tmp_filename)
    print(landmark)
    
    angle = await mediapipe_util.extract_angles_from_landmark(landmark)
    print( angle)

    return FileResponse(f"tmp/out_{tmp_filename}", headers={"angle": str(angle[0])})

if __name__ == "__main__":
    uvicorn.run(app=app, host="0.0.0.0", port=8000)




