#需要求加权平均值的数据列表
elements = [0.8306, 0.9266, 0.9037, 0.9339, 0.9023, 0.9243, 0.9173, 0.9002]
#对应的权值列表
weights = [4794, 5749, 5796, 5959, 5138, 5850, 6765, 6013]
a = 0
for i in weights:
    a = a+i
weights2 = []
for i in weights:
    m = i/a
    weights2.append(m)

import numpy as np

result = np.average(elements, weights=weights2)
print(result)