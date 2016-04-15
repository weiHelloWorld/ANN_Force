
# coding: utf-8

# In[1]:

from sympy import *
from sympy.physics.vector import *


# In[39]:

x11, x12, x13, x21, x22, x23, x31, x32, x33 = symbols('x11 x12 x13 x21 x22 x23 x31 x32 x33')

v1_squared, v2_squared = symbols('v1_squared v2_squared')
v1_x, v1_y, v1_z, v2_x, v2_y, v2_z = symbols('v1_x v1_y v1_z v2_x v2_y v2_z')

R = ReferenceFrame('R')

r1 = x11 * R.x + x12 * R.y + x13 * R.z
r2 = x21 * R.x + x22 * R.y + x23 * R.z
r3 = x31 * R.x + x32 * R.y + x33 * R.z

v1_ = cross(r1, r2)
v2_ = cross(r2, r3)

normal_1, normal_2 = symbols('normal_1 normal_2')
n_1x, n_1y, n_1z, n_2x, n_2y, n_2z = symbols('n_1x n_1y n_1z n_2x n_2y n_2z')

normal_1 = n_1x * R.x + n_1y * R.y + n_1z * R.z
normal_2 = n_2x * R.x + n_2y * R.y + n_2z * R.z

c = normal_1.dot(normal_2) / sqrt(normal_1.dot(normal_1) * normal_2.dot(normal_2))
s = sqrt(normal_1.cross(normal_2).dot(normal_1.cross(normal_2))) / sqrt(normal_1.dot(normal_1) * normal_2.dot(normal_2))


# In[65]:

print('RealOpenMM der_of_normal_to_diff[6][9];')

for index_1, item in enumerate([R.x,R.y,R.z]):
    for index_2, coor in enumerate([x11, x12, x13, x21, x22, x23, x31, x32, x33]):
        print("der_of_normal_to_diff[%d][%d] = %s;" %  ( index_1, index_2,                                                      str(simplify(diff(v1_.dot(item), coor)))))
for index_1, item in enumerate([R.x,R.y,R.z]):
    for index_2, coor in enumerate([x11, x12, x13, x21, x22, x23, x31, x32, x33]):
        print("der_of_normal_to_diff[%d][%d] = %s;" %  ( index_1 + 3, index_2,                                                      str(simplify(diff(v2_.dot(item), coor)))))


# In[64]:

temp = symbols('temp')

for index_2, element in enumerate([c, s]):
    for index, item in enumerate([n_1x, n_1y, n_1z, n_2x, n_2y, n_2z]):
        if element == s:
            print("der_of_cos_sin_to_nornal[%d][%d] = %s * sign;"                   %(index_2, index, str(simplify(diff(element, item)))))
        else:
            print("der_of_cos_sin_to_nornal[%d][%d] = %s;" %(index_2, index, str(simplify(diff(element, item)))))
    


# In[ ]:



