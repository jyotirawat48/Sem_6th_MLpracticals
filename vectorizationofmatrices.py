>>> import matplotlib.pyplot as plt
>>> plt.plot([1,2,3],[4,5,6])
[<matplotlib.lines.Line2D object at 0x00000224330A0790>]
>>> plt.draw()
>>> plt.show()

VECTORIZATION:
>>> import numpy as np
>>> #converting a tuple into a vector
>>> #using numpy methods to convert into a array
>>> #1-D array
>>> type((1,2,3,4,5))
<class 'tuple'>
>>> x=np.array((1,2,3,4,5))
>>> x
array([1, 2, 3, 4, 5])
>>> type(x)
<class 'numpy.ndarray'>
>>> #2-D array
>>> y=np.random.randn(3,4)
>>> y
array([[ 0.73970731, -0.11279598,  0.26243823, -0.77851299],
       [-0.53945554, -0.95227891, -0.41367667,  0.08030448],
       [-0.83408442, -0.99784631, -1.91956168,  1.04841069]])
>>> type(y)
<class 'numpy.ndarray'>
>>> #2-D array -->matrices
>>>  

OPERATIONS ON VECTORS:
>>> import numpy as np
>>> x=np.array([1,2,3])
>>> y=np.array([9,8,7])
>>> #addition
>>> x+y
array([10, 10, 10])
>>> #subtraction
>>> y-x
array([8, 6, 4])
>>> #product
>>> x*y
array([ 9, 16, 21])

OPERATIONS ON MATRICES:
>>> mat1=np.random.randint(1,10,(3,3))
>>> mat2=np.random.randint(1,10,(3,3))
>>> mat1
array([[6, 3, 8],
       [3, 4, 5],
       [9, 3, 6]])
>>> mat2
array([[3, 7, 6],
       [6, 8, 6],
       [6, 7, 4]])
>>> #addition
>>> mat1+mat2
array([[ 9, 10, 14],
       [ 9, 12, 11],
       [15, 10, 10]])
>>> #subtration
>>> mat1-mat2
array([[ 3, -4,  2],
       [-3, -4, -1],
       [ 3, -4,  2]])
>>> #product
>>> mat1*mat2
array([[18, 21, 48],
       [18, 32, 30],
       [54, 21, 24]])
>>> #matrix multipliaction
>>> mat1.dot(mat2)
array([[ 84, 122,  86],
       [ 63,  88,  62],
       [ 81, 129,  96]])
>>> #transpose
>>> np.transpose(mat1)
array([[6, 3, 9],
       [3, 4, 3],
       [8, 5, 6]])
>>> np.transpose(mat2)
array([[3, 6, 6],
       [7, 8, 7],
       [6, 6, 4]])
>>> #inverse
>>> np.linalg.inv(mat1)
array([[-0.11111111, -0.07407407,  0.20987654],
       [-0.33333333,  0.44444444,  0.07407407],
       [ 0.33333333, -0.11111111, -0.18518519]])
>>> np.linalg.inv(mat2)
array([[-0.55555556,  0.77777778, -0.33333333],
       [ 0.66666667, -1.33333333,  1.        ],
       [-0.33333333,  1.16666667, -1.        ]])
>>>      
