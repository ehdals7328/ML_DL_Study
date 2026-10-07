# 프로그래밍 문제 2: YOLO 클래스별 신뢰도와 최종 판정
import numpy as np

conf = np.array([0.9, 0.3, 0.8])
pred = np.array([
    [0.80, 0.15, 0.05],
    [0.10, 0.20, 0.70],
    [0.20, 0.70, 0.10]
])

classes = ['자동차', '자전거', '보행자']
THRESHOLD = 0.5

conf = conf.reshape(3, 1)
calc = conf * pred # 브로드캐스팅 한 점수 행렬

max_score = np.max(calc, axis=1)
max_index = np.argmax(calc, axis=1)

print(max_score)
print(max_index)
print(" === 결과 === ")
for i in range(len(conf)):
    score = max_score[i]
    name = classes[max_index[i]]

    if score >= THRESHOLD:
        print(f"박스 {i}: {name} ({score:.2f})")

    else:
        print(f"박스 {i}: 제거")