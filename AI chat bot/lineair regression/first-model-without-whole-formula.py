import numpy as np # linear algebra
import matplotlib.pyplot as plt

xrange = np.linspace(100,0,100)
def model(feature):
    return 65.14 + (0.385225 * feature)
x = [43,21,25,42,57,59, 60, 40, 80, 10, 30, 65, 50]
y = [99,65,79,75,87,81, 85, 75, 90, 67, 80, 90, 90]
m = len(y)

plt.scatter(x,y)
plt.plot(xrange, model(xrange), '-r', label='linear regression')
plt.show()

print(model(18))
