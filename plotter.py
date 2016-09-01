from math import sqrt
import numpy as np
from threading import Thread
from matplotlib import pyplot as plt
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from time import sleep
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
from sympy import Symbol, symbols, Matrix, sin, cos, sqrt, atan
from sympy import init_printing
_FLOAT_EPS_4 = np.finfo(float).eps * 4.0
init_printing(use_latex=True)
dt = 0.1

m = 1030.0 	# Mass of vehicle in kg
ms = 810.0	# Kg
h = 0.505 	# m
hs = 0.22
tf = 1.28	# m
tr = tf
d = tf/2
a = 0.968	# m
b = 1.392	# m
wheelr = 0.22
ks_FL = 20600.0       # Spring Constant N/m
ks_FR = 20600.0
ks_RL = 15200.0
ks_RR = 15200.0
c_FL = 1570.0        # Damping Constant N.s/m
c_FR = 1570.0
c_RL = 1760.0
c_RR = 1760.0
m_b = 1.0         # Mass of the vehicle sans frong and rear wheels
m_wFL = 25.0       # Mass of wheel
m_wFR = 25.0
m_wRL = 25.0
m_wRR = 25.0
kaf = 6695.0
g = 9.81
dof = 14.0

Ix = 300.0
Iy = 1058.4
Iz = 1087.8

Ca_FL = 200520.0
Ca_FR = 200520.0
Ca_RL = 76426.0
Ca_RR = 76426.0

Ck_FL = 30.0
Ck_FR = 30.0
Ck_RL = 24.0
Ck_RR = 24.0

kroll = 42075.0
croll = 5737.5

wheelr = (0.4/2.0) + 0.11

import warnings
warnings.filterwarnings('ignore')

def mat2euler(M, cy_thresh=None):
    M = np.asarray(M)
    if cy_thresh is None:
        try:
            cy_thresh = np.finfo(M.dtype).eps * 4
        except ValueError:
            cy_thresh = _FLOAT_EPS_4
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
    cy = math.sqrt(r33*r33 + r23*r23)
    if cy > cy_thresh: # cos(y) not close to zero, standard form
        z = math.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
        y = math.atan2(r13,  cy) # atan2(sin(y), cy)
        x = math.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))
    else: # cos(y) (close to) zero, so x -> 0.0 (see above)
        # so r21 -> sin(z), r22 -> cos(z) and
        z = math.atan2(r21,  r22)
        y = math.atan2(r13,  cy) # atan2(sin(y), cy)
        x = 0.0
    return z, y, x

x = [line.strip() for line in open("VO_data_x.txt", 'r')]
y = [line.strip() for line in open("VO_data_y.txt", 'r')]
z = [line.strip() for line in open("VO_data_z.txt", 'r')]

datafile = '/home/equaltrace/catkin_ws/src/viso2-indigo/viso2_ros/data/Statedata1.csv'
x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30,x31,x32,x33,x34,x35,x36,x37,x38,x39,x40,x41,x42,x43,x44,x45,x46,x47,x48,x49= np.loadtxt(datafile, delimiter=',', unpack=True)
print('Read \'%s\' successfully.' % datafile)

datafile = "/home/equaltrace/catkin_ws/src/viso2-indigo/viso2_ros/scripts/03.csv"

dataleng = len(x0)

row1_e1, row1_e2, row1_e3, row1_e4, row2_e1, row2_e2, row2_e3, row2_e4, row3_e1, row3_e2, row3_e3, row3_e4 = np.loadtxt(datafile, delimiter=',', unpack=True)


print('Read \'%s\' successfully.' % datafile)


iterLen = len(row1_e1)
yaw_meas_S = []
pitch_meas_S = []
roll_meas_S = []
x_meas = row3_e4
y_meas = -row1_e4
z_meas = -row2_e4

for i in range(0, iterLen):
    inMat = [[row1_e1[i], row1_e2[i], row1_e3[i]], [row2_e1[i], row2_e2[i], row2_e3[i]], [row3_e1[i], row3_e2[i], row3_e3[i]]]
    zYaw, yPitch, xRoll = mat2euler(inMat)
    yaw_meas_S.append(zYaw)
    pitch_meas_S.append(yPitch)
    roll_meas_S.append(xRoll)

velX = np.zeros(len(x_meas))
velY = np.zeros(len(x_meas))
velZ = np.zeros(len(x_meas))
velYaw = np.zeros(len(x_meas))
velPitch = np.zeros(len(x_meas))
velRoll = np.zeros(len(x_meas))

x2 = y_meas[0:dataleng] + np.random.normal(0.7,1,dataleng)


x1 = velX[0:dataleng]+ np.random.normal(0.1,0.1,dataleng)
x3 = velY[0:dataleng]+ np.random.normal(0.1,0.1,dataleng)

velX[0] = 0.0
velY[0] = 0.0
velZ[0] = 0.0
velYaw[0] = 0.0
velPitch[0] = 0.0
velRoll[0] = 0.0

for i in range(1,len(x_meas)):
    velX[i] = x_meas[i] - x_meas[i-1]
    velY[i] = y_meas[i] - y_meas[i-1]
    velZ[i] = z_meas[i] - z_meas[i-1]
    velYaw[i] = pitch_meas_S[i] - pitch_meas_S[i -1]
    velPitch[i] = roll_meas_S[i] - roll_meas_S[i - 1]
    velRoll[i] = yaw_meas_S[i] - yaw_meas_S[i - 1]

acclX = np.zeros(len(x_meas))
acclY = np.zeros(len(x_meas))
acclZ = np.zeros(len(x_meas))
acclYaw = np.zeros(len(x_meas))
acclPitch = np.zeros(len(x_meas))
acclRoll = np.zeros(len(x_meas))

acclX[0] = 0.0
acclY[0] = 0.0
acclZ[0] = 0.0
acclYaw[0] = 0.0
acclPitch[0] = 0.0
acclRoll[0] = 0.0

for i in range(1,len(x_meas)):
    acclX[i] = velX[i] - velX[i-1]
    acclY[i] = velY[i] - velY[i-1]
    acclZ[i] = velZ[i] - velZ[i-1]
    acclYaw[i] = velYaw[i] - velYaw[i -1]
    acclPitch[i] = velPitch[i] - velPitch[i - 1]
    acclRoll[i] = velRoll[i] - velRoll[i - 1]

beta= np.zeros(len(x_meas))

beta[0]= 0.0
for i in range(1, len(x_meas)):
    try:
        beta[i] = atan(velY[i]/velX[i])
    except RuntimeWarning:
        beta[i] = 0

fzFL = np.zeros(len(x_meas))
fzFR = np.zeros(len(x_meas))
fzRL = np.zeros(len(x_meas))
fzRR = np.zeros(len(x_meas))

fzFL[0] = 0.0
fzFR[0] = 0.0
fzRL[0] = 0.0
fzRR[0] = 0.0

for i in range(1, len(x_meas)):
    fzFL[i] = (0.5*m*g + ((m*h*acclY[i])/tf))*(b/(a+b)) - 0.5*m*acclX[i]*(h/(a+b))#;       # 26. fzFL
    fzFL[i] = (0.5*m*g - ((m*h*acclY[i])/tf))*(b/(a+b)) - 0.5*m*acclX[i]*(h/(a+b))#;       # 27. fzFR
    fzFL[i] = (0.5*m*g + ((m*h*acclY[i])/tr))*(a/(a+b)) + 0.5*m*acclX[i]*(h/(a+b))#;       # 28. fzRL
    fzFL[i] = (0.5*m*g - ((m*h*acclY[i])/tr))*(a/(a+b)) + 0.5*m*acclX[i]*(h/(a+b))#;       # 29. fzRR

actwV_FL = np.zeros(len(x_meas))
actwV_FR = np.zeros(len(x_meas))
actwV_RL = np.zeros(len(x_meas))
actwV_RR = np.zeros(len(x_meas))

actwV_FL[0] = 0.0
actwV_FR[0] = 0.0
actwV_RL[0] = 0.0
actwV_RR[0] = 0.0


for i in range(1, len(x_meas)):
    actwV_FL[i] = sqrt(velX[i]**2 + velY[i]**2) + velYaw[i]*((tf/2)-a*beta[i])#;    # 38. FL acutal wheel velocity
    actwV_FR[i] = sqrt(velX[i]**2 + velY[i]**2) + velYaw[i]*((-tf/2)-a*beta[i])#;   # 39. FR
    actwV_RL[i] = sqrt(velX[i]**2 + velY[i]**2) + velYaw[i]*((tf/2)+b*beta[i])#;    # 40. RL
    actwV_RR[i] = sqrt(velX[i]**2 + velY[i]**2) + velYaw[i]*((-tf/2)+b*beta[i])#;   # 41. RR


plt.figure(1)

plt.subplot(521)
plt.title('x')
plt.plot(range(dataleng),x0)
plt.plot(range(dataleng),x_meas[0:dataleng])

plt.subplot(522)
plt.title('y')
plt.plot(range(dataleng),x2)
plt.plot(range(dataleng),y_meas[0:dataleng])

plt.subplot(523)
plt.title('xVel')
plt.plot(range(dataleng),x1)
plt.plot(range(dataleng),velX[0:dataleng])

plt.subplot(524)
plt.title('yVel')
plt.plot(range(dataleng),x3)
plt.plot(range(dataleng),velY[0:dataleng])

plt.subplot(525)
plt.title('yaw')
plt.plot(range(dataleng),x4)
plt.plot(range(dataleng),pitch_meas_S[0:dataleng])

plt.subplot(526)
plt.title('yawRate')
plt.plot(range(dataleng),x5)
plt.plot(range(dataleng),velYaw[0:dataleng])

plt.subplot(527)
plt.title('roll')
plt.plot(range(dataleng),x6)
plt.plot(range(dataleng),yaw_meas_S[0:dataleng])

plt.subplot(528)
plt.title('rollRate')
plt.plot(range(dataleng),x7)
plt.plot(range(dataleng),velRoll[0:dataleng])

plt.subplot(529)
plt.title('pitch')
plt.plot(range(dataleng),x8)
plt.plot(range(dataleng),roll_meas_S[0:dataleng])

plt.subplot(5,2,10)
plt.title('pitchR')
plt.plot(range(dataleng),x9)
plt.plot(range(dataleng),velPitch[0:dataleng])

plt.show()
