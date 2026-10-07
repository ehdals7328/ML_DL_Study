import numpy as np
from collections import Counter

vehicles = np.array([
    [0.3, 0.2, 0.4, 0.7],     
    [0.4, 0.25, 0.3, 0.8],    
    [0.35, 0.22, 0.35, 0.75], 
    [0.8, 0.7, 0.6, 0.4],     
    [0.85, 0.75, 0.5, 0.3],   
    [0.82, 0.72, 0.55, 0.35], 
    [0.7, 0.6, 0.8, 0.2],     
    [0.75, 0.65, 0.85, 0.15], 
    [0.8, 0.7, 0.9, 0.1],     
    [0.5, 0.5, 0.6, 0.5],     
    [0.45, 0.48, 0.65, 0.55], 
    [0.52, 0.51, 0.58, 0.52]  
])

classes = np.array([
    "승용차", "승용차", "승용차",
    "버스", "버스", "버스",
    "트럭", "트럭", "트럭",
    "SUV", "SUV", "SUV"
])

new_car = np.array([0.55, 0.45, 0.62, 0.48])

def distance(a, b): # 유클리디안 디스턴스
    dist = np.zeros(12)
    for i in range(12):
        dist[i] = np.sqrt(np.sum((a[i] - b) ** 2))

    return dist 

def majority_vote(dist, classes, k=3):
    nst = np.argsort(dist)[:k]

    nst_class = classes[nst]

    majority = Counter(nst_class)

    return majority

dist = distance(vehicles, new_car)
maj = majority_vote(dist, classes, k=3)

print("=== k=3 최종 다수결 결과 ===")
print(maj)
print("=== 최종 분류 (최다 득표) ===")
print(maj.most_common(1)[0][0])