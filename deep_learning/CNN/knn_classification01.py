# 프로그래밍 문제 1 k-NN 기반 차량 분류
import numpy as np

vehicle = np.array([
    [0.8, 0.7, 0.6, 0.4],
    [0.7, 0.6, 0.8, 0.2],
    [0.5, 0.5, 0.6, 0.5]
])

new_car = np.array([0.55, 0.45, 0.62, 0.48])

classes = ["버스", "트럭", "SUV "]

def distance(a, b): # 유클리디안 디스턴스
    dist = np.zeros(3)
    for i in range(3):
        dist[i] = np.sqrt(np.sum((a[i] - b) ** 2))

    return dist 

def find_nn(dist):
    nearest = np.argmin(dist)

    if nearest < len(classes):
        print(classes[nearest])
    else:
        print("clsss 정의 오류")
    return nearest

result = distance(vehicle, new_car) # classes 와 분류할 차의 유클리디안 거리
answer = find_nn(result)

print(" === Euclidean distance === ")
for i, object in enumerate(classes):
    print(f"{object}: {result[i]:.5f}")
print(" === 새 차량과 가장 가까운 class === ")
print(f"{classes[answer]}")