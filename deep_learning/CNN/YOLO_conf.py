# 프로그래밍 문제 2: YOLO 클래스별 신뢰도와 최종 판정
import numpy as np

conf = np.array([0.9, 0.3, 0.8])

pred = np.array([
    [0.80, 0.15, 0.05],
    [0.10, 0.20, 0.70],
    [0.20, 0.70, 0.10]
]) # 자동차 / 자전거 / 보행자

result = np.zeros(3) # 각 box 별 최댓값을 저장
calc = conf * pred # 브로드캐스팅 한 점수 행렬
THRESHOLD = 0.5

for i in range(3):
    result[i] = np.max(calc[i])

for i in range(3):
    if result[i] < THRESHOLD:
        result[i] = None

print(calc)
print(result)