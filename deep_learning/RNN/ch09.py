# 문제 : Vanilla RNN의 순전파 (Forward Propagation) 알고리즘 구현하기
import numpy as np

u = np.random.rand(4, 5)
v = np.random.rand(2, 4)
w = np.random.rand(4, 4)
h = np.zeros((4,1)) # 초기 h0는 zero 로 설정
x = np.random.rand(5, 5, 1) # 입력 차원

ht_list = []
ot_list = []

def ht(u, w, h, x):
    return (u @ x) + (w @ h) # activation f 통과전

def ot(v, h):
    return (v @ h)

def f_tanh(h_out):
    return np.tanh(h_out)

def g_sigmoid(x): # output 용 sigmoid
    return 1 / (1 + np.exp(-x))

print(" === 바닐라 RNN 순전파 테스트 결과 === ")

for i in range(len(x)):

    h = (f_tanh(ht(u, w, h, x[i]))) # memo
    ht_list.append(h)

    o = (g_sigmoid(ot(v, h)))
    ot_list.append(o)

    print(f"타임스텝 t = {i+1}")
    print("[은닉 상태 ht]")
    print(h)
    print("[출력값 ot]")
    print(o)
    print("-" * 15)



"""
==========================================
 ot 출력 층 (2,1) / ht 출력층 (4,1)
 ot = V * ht 이므로 v = (2,4)
      ht  =   U  * xt   +   W   * ht-1
    (4,1) = (4,i)*(i,1) + (4,4) * (4,1)
i = 5 (x1 ~ x5)
------------------------------------------
# time step 1
h2 = f_tanh(ht(u, w, h, x[1]))
o1 = g_softmax(ot(v, h2))

# time step 2
h3 = f_tanh(ht(u, w, h2, x[2]))
o2 = g_softmax(ot(v, h3))

#time step 3
h4 = f_tanh(ht(u, w, h3, x[3]))
o3 = g_softmax(ot(v, h4))

#time step 4
h5 = f_tanh(ht(u, w, h4, x[4]))
o4 = g_softmax(ot(v, h5))

#time step 5
h6 = f_tanh(ht(u, w, h5, x[5]))
o5 = g_softmax(ot(v, h6))
------------------------------------------
ex) x = I like chicken
x[0] = I
x[1] = like
x[2] = chicken

d1 = Vector
d2 = Matrix
d3 = 3D Tensor 

(3,) word = [0.1 0.5 0.3]

(3, 3) sentence = [[0.1 0.5 0.3],
                   [0.2 0.4 0.2],
                   [0.1 0.8 0.9]]

(3, 3, 3) data = np.array([
            # 문장 1 ("I like chicken")
            [[0.1, 0.5, 0.3],
            [0.2, 0.4, 0.2],
            [0.1, 0.8, 0.9]],

            # 문장 2 ("He hates pizza")
            [[0.1, 0.5, 0.3],
            [0.2, 0.4, 0.2],
            [0.1, 0.8, 0.9]],

            # 문장 3 ("We drink water")
            [[0.1, 0.5, 0.3],
            [0.2, 0.4, 0.2],
            [0.1, 0.8, 0.9]]
        ])
==========================================
"""
