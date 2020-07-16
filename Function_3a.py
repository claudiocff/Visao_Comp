import numpy as np
def mean(x):
    s = 0
    for num in x:
        s += num
    mean = s / len(x)
    return mean
'''
print(" ")
print("Exemple")
print(" ")
x = np.array([1, 2, 3])
print(mean(x))
print(" ")
'''
def var(x):
    y = 0
    m = mean(x)
    N = len(x)
    for num in x:
       y = y + ((num-m)**2)/N
    return y
'''
print(" ")

print("Exemple")
print(" ")

print(var(x))
'''