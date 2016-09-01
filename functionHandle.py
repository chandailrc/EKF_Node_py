import csv
from itertools import izip
from math import sqrt
#import numpy as np, rospy
from threading import Thread
from matplotlib import pyplot as plt
#from geometry_msgs.msg import PoseStamped
#from nav_msgs.msg import Odometry
from time import sleep
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
from sympy import *
import sys
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
from sympy import Symbol, symbols, Matrix, sin, cos, sqrt, atan, im
from sympy import init_printing
_FLOAT_EPS_4 = np.finfo(float).eps * 4.0
init_printing(use_latex=True)

import warnings
warnings.filterwarnings('ignore')



dt = 0.1

m = 1030.0 	# Mass of vehicle in kg
ms = 810.0	# Kg
h = 0.505 	# m
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

wheelr = (0.4/2.0) + 0.11

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

def x_handle(x):
    x[0] = x[-1+1] + x[-1+2] * dt + 0.5 * x[-1+14] * (dt ** 2)#;  # 1. x
    x[1] = x[-1+2] + x[-1+14] * dt#;  # 2. x vel
    x[2] = x[-1+3] + x[-1+4] * dt + 0.5 * x[-1+15] * (dt ** 2)#;  # 3.z
    x[3] = x[-1+4] + x[-1+15] * dt#;  # 4. z vel
    x[4] = x[-1+5] + x[-1+6] * dt + 0.5 * x[-1+47] * (dt ** 2)#;  # 5. yaw
    x[5] = x[-1+6] + x[-1+47] * dt#;  # 6. yaw rate
    x[6] = x[-1+7] + x[-1+8] * dt + 0.5 * x[-1+49] * (dt ** 2)#;  # 7. roll
    x[7] = x[-1+8] + x[-1+49] * dt#;  # 8. roll rate
    x[8] = x[-1+9] + x[-1+10] * dt + 0.5 * x[-1+48] * (dt ** 2)#;  # 9. pitch
    x[9] = x[-1+10] + x[-1+48] * dt#;  # 10. pitch rate
    x[10] = x[-1+11] + x[-1+12] * dt + 0.5 * x[-1+16] * (dt ** 2)#;  # 11. y
    x[11] = x[-1+12] + x[-1+16] * dt#;  # 12. y vel
    x[12] = x[-1+13] + x[-1+17]*dt#;       # 13. Vxy
    x[13] = x[-1+14]#; # (1/m)*((x[-1+18]+x[-1+19])*cos(x[-1+50]) - (x[-1+22]+x[-1+23])*sin(x[-1+50]) + x[-1+20] + x[-1+21] + ms*h*x[-1+47]*x[-1+7]) + x[-1+12]*x[-1+5]#;      # 14. ax
    x[14] = x[-1+15]#; # 15. az
    x[15] = x[-1+16]#; # (1/m)*((x[-1+22]+x[-1+23])*cos(x[-1+50]) + (x[-1+18]+x[-1+19])*sin(x[-1+50]) + x[-1+24] + x[-1+25] - ms*h*x[-1+49]) - x[-1+2]*x[-1+5]#;      # 16. ay
    x[16] = x[-1+17]#; # 17. Axy
    x[17] = x[-1+18]#; #-Ck_FL * x[-1+30]#;       # 18. fxFL
    x[18] = x[-1+19]#; #-Ck_FR * x[-1+31]#;       # 19. fxFR
    x[19] = x[-1+20]#; #-Ck_RL * x[-1+32]#;       # 20. fxRL
    x[20] = x[-1+21]#; #-Ck_RR * x[-1+33]#;       # 21. fxRR
    x[21] = x[-1+22]#; #-Ca_FL * x[-1+43]#;       # 22. fyFL
    x[22] = x[-1+23]#; #-Ca_FR * x[-1+44]#;       # 23. fyFR
    x[23] = -Ca_RL * x[-1+45]#;       # 24. fyRL
    x[24] = -Ca_RR * x[-1+46]#;       # 25. fyRR
    #x[25] = #ks_FL*(-x[-1+54]) + c_FL*(-x[-1+58]) + m_wFL*g +
    x[25] = (0.5*m*g + ((m*h*x[-1+16])/tf))*(b/(a+b)) - 0.5*m*x[-1+14]*(h/(a+b))#;       # 26. fzFL
    #x[26] = #ks_FR*(-x[-1+55]) + c_FR*(-x[-1+59]) + m_wFR*g +
    x[26] = (0.5*m*g - ((m*h*x[-1+16])/tf))*(b/(a+b)) - 0.5*m*x[-1+14]*(h/(a+b))#;       # 27. fzFR
    #x[27] = #ks_RL*(-x[-1+56]) + c_RL*(-x[-1+60]) + m_wRL*g +
    x[27] = (0.5*m*g + ((m*h*x[-1+16])/tr))*(a/(a+b)) + 0.5*m*x[-1+14]*(h/(a+b))#;       # 28. fzRL
    #x[28] = #ks_RR*(-x[-1+57]) + c_RR*(-x[-1+61]) + m_wRR*g +
    x[28] = (0.5*m*g - ((m*h*x[-1+16])/tr))*(a/(a+b)) + 0.5*m*x[-1+14]*(h/(a+b))#;       # 29. fzRR
    x[29] = x[-1+30]#; #x[-1+34]*wheelr/x[-1+38] - 1#;      # 30. slip ratio FL
    x[30] = x[-1+31]#; #x[-1+35]*wheelr/x[-1+39] - 1#;      # 31. slip ratio FR
    x[31] = x[-1+32]#; #x[-1+36]*wheelr/x[-1+40] - 1#;      # 32. slip ratio RL
    x[32] = x[-1+33]#; #x[-1+37]*wheelr/x[-1+41] - 1#;      # 33. slip ratio RR
    x[33] = x[-1+34]#;       # 34. wheel angular velocity FL
    x[34] = x[-1+35]#;       # 35. FR
    x[35] = x[-1+36]#;       # 36. RL
    x[36] = x[-1+37]#;       # 37. RR
    x[37] = sqrt(x[-1+2]**2 + x[-1+12]**2) + x[-1+6]*((tf/2)-a*x[-1+42])#;    # 38. FL acutal wheel velocity
    x[38] = sqrt(x[-1+2]**2 + x[-1+12]**2) + x[-1+6]*((-tf/2)-a*x[-1+42])#;   # 39. FR
    x[39] = sqrt(x[-1+2]**2 + x[-1+12]**2) + x[-1+6]*((tf/2)+b*x[-1+42])#;    # 40. RL
    x[40] = sqrt(x[-1+2]**2 + x[-1+12]**2) + x[-1+6]*((-tf/2)+b*x[-1+42])#;   # 41. RR
    x[41] = atan(x[-1+12]/x[-1+2])#;        # 42. Beta
    x[42] = x[-1+43]#; # 43. x[-1+50] - atan((x[-1+12] + b*x[-1+6])/(x[-1+2] + tf*x[-1+6]*0.5))#; # alpha FL
    x[43] = x[-1+44]#; # 44. x[-1+50] - atan((x[-1+12] + b*x[-1+6])/(x[-1+2] - tf*x[-1+6]*0.5))#; # alpha FR
    x[44] = atan((-x[-1+12] + b*x[-1+6])/(x[-1+2] + tr*x[-1+6]*0.5))#;        # 45. alpha RL
    x[45] = atan((-x[-1+12] + b*x[-1+6])/(x[-1+2] - tr*x[-1+6]*0.5))#;        # 46. alpha RR
    x[46] = x[-1+47]#; #(1/Iz)*(tf*0.5*(x[-1+18]*cos(x[-1+50]) - x[-1+22]*sin(x[-1+50]) - x[-1+19]*cos(x[-1+50]) + x[-1+23]*sin(x[-1+50])) + tr*0.5*(x[-1+20]-x[-1+21]) + a*(x[-1+22]*cos(x[-1+50])+x[-1+18]*sin(x[-1+50])+x[-1+23]*cos(x[-1+50])+x[-1+19]*sin(x[-1+50])) - b*(x[-1+24]-x[-1+25]))#;      # 47. yawA
    x[47] = x[-1+48]#; #(1/Iy)*(b*(x[-1+29]+x[-1+28]) - a*(x[-1+27]+x[-1+26]))#;  # 48. pitchA
    x[48] = x[-1+49]#; #(1/Ix)*(ms*hs*(-x[-1+16]+g*x[-1+7]) - kroll*x[-1+7] - croll*x[-1+8])#; #ms*g*h*x[-1+7] - ms*h*(x[-1+16] + x[-1+14]*x[-1+8]) + (x[-1+26]+x[-1+29]-x[-1+28]-x[-1+27])*d#;       # 49. rollA
    x[49] = x[-1+50]#; # 50. delta
    return x

def y_handle(x):

    y = []
    y.append(x[0])
    y.append(x[10])
    y.append(x[2])
    y.append(x[4])
    y.append(x[8])
    y.append(x[6])
    y.append(x[0])
    y.append(x[10])
    y.append(x[2])
    return y

n_states = 50
n_meas_states = 9

def imagiN(x1Pass):
    x1Ret = []
    for x2 in range(0, len(x1Pass)):
        if isinstance(x1Pass[x2], Float) or isinstance(x1Pass[x2], Add):
            x1Ret.append(im(x1Pass[x2]))
        else:
            x1Ret.append(x1Pass[x2].imag)
    return x1Ret


def jaccsd(fun, x):
    z = fun(x)
    n = len(x)
    m = len(z)
    A = np.zeros((m, n))
    h = n * np.finfo(float).eps

    for k in range(0, m):
        x1 = np.zeros(n,dtype=complex)
        x1 = x1 + x
        x1[k] = x1[k] + h * 1j
        A[:, k] = imagiN(fun(x1))/h;

    return z, A

#xi = []
#for i in range(50):
#    xi.append(1.0)

#print(y_handle(xi))

# EKF

def writerf(x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30,x31,x32,x33,x34,x35,x36,x37,x38,x39,x40,x41,x42,x43,x44,x45,x46,x47,x48,x49, datafile1):

    dt = 0.1

    m = 1030.0 	# Mass of vehicle in kg
    ms = 810.0	# Kg
    h = 0.505 	# m
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

    wheelr = (0.4/2.0) + 0.11

    datafile = '/home/equaltrace/catkin_ws/src/viso2-indigo/viso2_ros/scripts/03.csv'
    row1_e1, row1_e2, row1_e3, row1_e4, row2_e1, row2_e2, row2_e3, row2_e4, row3_e1, row3_e2, row3_e3, row3_e4 = np.loadtxt(datafile, delimiter=',', unpack=True)
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

    velX[0] = 0.0
    velY[0] = 0.0
    velZ[0] = 0.0
    velYaw[0] = 0.0
    velPitch[0] = 0.0
    velRoll[0] = 0.0

    for i in range(1,len(x_meas)):
        velX[i] = (x_meas[i] - x_meas[i-1])
        velY[i] = (y_meas[i] - y_meas[i-1])
        velZ[i] = (z_meas[i] - z_meas[i-1])
        velYaw[i] = (pitch_meas_S[i] - pitch_meas_S[i -1])
        velPitch[i] = (roll_meas_S[i] - roll_meas_S[i - 1])
        velRoll[i] = (yaw_meas_S[i] - yaw_meas_S[i - 1])

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
        acclX[i] = (velX[i] - velX[i-1])
        acclY[i] = (velY[i] - velY[i-1])
        acclZ[i] = (velZ[i] - velZ[i-1])
        acclYaw[i] = (velYaw[i] - velYaw[i -1])
        acclPitch[i] = (velPitch[i] - velPitch[i - 1])
        acclRoll[i] = (velRoll[i] - velRoll[i - 1])

    beta = np.zeros(len(x_meas))
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

    sinS = np.zeros(len(x_meas))
    sinA = np.zeros(len(x_meas))
    cosS = np.zeros(len(x_meas))
    cosA = np.zeros(len(x_meas))


    for i in range(1, len(x_meas)):
        sinS[i] = 2.5*sin(i*math.pi/100)
        sinA[i] = 0.01*sin(i*math.pi/100)
        cosS[i] = 2.5*cos(i*math.pi/100)
        cosA[i] = 0.03*cos(i*math.pi/100)

    x0 = x_meas[0:len(x0)] + np.random.normal(0.5,0.5,len(x0)) + sinS[0:len(x0)]# 1. x
    x1 = velX[0:len(x0)] + np.random.normal(0.5,0.06,len(x0))  + 0.1*cosS[0:len(x0)]  #;  # 2. x vel
    x2 = z_meas[0:len(x0)] + np.random.normal(0.7,0.6,len(x0)) + sinS[0:len(x0)]#;  # 3.z
    x3 = velZ[0:len(x0)] + np.random.normal(1,0.06,len(x0))+ 0.1*cosS[0:len(x0)]#;  # 4. z vel
    x4 = pitch_meas_S[0:len(x0)]+ np.random.normal(0.02,0.0086,len(x0))+ cosA[0:len(x0)]#;  # 5. yaw
    x5 = velYaw[0:len(x0)]+ np.random.normal(0.03,0.0085,len(x0)) + sinA[0:len(x0)]#;  # 6. yaw rate
    x6 = yaw_meas_S[0:len(x0)]+ np.random.normal(0.05,0.0086,len(x0)) + cosA[0:len(x0)]#;  # 7. roll
    x7 = velRoll[0:len(x0)]+ np.random.normal(0.07,0.0084,len(x0)) + sinA[0:len(x0)]#;  # 8. roll rate
    x8 = roll_meas_S[0:len(x0)]+ np.random.normal(0.05,0.0083,len(x0)) + cosA[0:len(x0)]#;  # 9. pitch
    x9 = velPitch[0:len(x0)]+ np.random.normal(0.05,0.0085,len(x0)) + sinA[0:len(x0)]#;  # 10. pitch rate
    x10 = y_meas[0:len(x0)]+ np.random.normal(0.5,0.06,len(x0)) + sinS[0:len(x0)]#;  # 11. y
    x11 = velY[0:len(x0)]+ np.random.normal(0.5,0.061,len(x0)) + 0.1*cosS[0:len(x0)]#;  # 12. y vel
    x13 = acclX[0:len(x0)]+ np.random.normal(0.5,0.61,len(x0)) + sinS[0:len(x0)]#; # (1/m)*((x[-1+18]+x[-1+19])*cos(x[-1+50]) - (x[-1+22]+x[-1+23])*sin(x[-1+50]) + x[-1+20] + x[-1+21] + ms*h*x[-1+47]*x[-1+7]) + x[-1+12]*x[-1+5]#;      # 14. ax
    x14 = acclZ[0:len(x0)]+ np.random.normal(0.5,0.61,len(x0)) + sinS[0:len(x0)]#; # 15. az
    x15 = acclY[0:len(x0)]+ np.random.normal(0.5,0.61,len(x0)) + sinS[0:len(x0)]#; # (1/m)*((x[-1+22]+x[-1+23])*cos(x[-1+50]) + (x[-1+18]+x[-1+19])*sin(x[-1+50]) + x[-1+24] + x[-1+25] - ms*h*x[-1+49]) - x[-1+2]*x[-1+5]#;      # 16. ay
    x25 = fzFL[0:len(x0)]+ np.random.normal(-0.6,0.61,len(x0)) + cosS[0:len(x0)]#;       # 26. fzFL
    x26 = fzFR[0:len(x0)]+ np.random.normal(-0.6,0.61,len(x0)) + cosS[0:len(x0)]#;       # 27. fzFR
    x27 = fzRL[0:len(x0)]+ np.random.normal(-0.6,0.61,len(x0)) + cosS[0:len(x0)]#;       # 28. fzRL
    x28 = fzRR[0:len(x0)]+ np.random.normal(-0.6,0.61,len(x0)) + cosS[0:len(x0)]#;       # 29. fzRR
    x37 = actwV_FL[0:len(x0)]+ np.random.normal(0.6,0.61,len(x0)) + sinS[0:len(x0)]#;    # 38. FL acutal wheel velocity
    x38 = actwV_FR[0:len(x0)]+ np.random.normal(0.6,0.61,len(x0)) + sinS[0:len(x0)]#;   # 39. FR
    x39 = actwV_RL[0:len(x0)]+ np.random.normal(0.64,0.61,len(x0)) + sinS[0:len(x0)]#;    # 40. RL
    x40 = actwV_RR[0:len(x0)]+ np.random.normal(0.61,0.61,len(x0)) + sinS[0:len(x0)]#;   # 41. RR
    x41 = beta[0:len(x0)]+ np.random.normal(0.09,0.0085,len(x0)) + cosA[0:len(x0)]#;        # 42. Beta
    x46 = acclYaw[0:len(x0)]+ np.random.normal(0.05,0.0086,len(x0)) + cosA[0:len(x0)]#; #(1/Iz)*(tf*0.5*(x[-1+18]*cos(x[-1+50]) - x[-1+22]*sin(x[-1+50]) - x[-1+19]*cos(x[-1+50]) + x[-1+23]*sin(x[-1+50])) + tr*0.5*(x[-1+20]-x[-1+21]) + a*(x[-1+22]*cos(x[-1+50])+x[-1+18]*sin(x[-1+50])+x[-1+23]*cos(x[-1+50])+x[-1+19]*sin(x[-1+50])) - b*(x[-1+24]-x[-1+25]))#;      # 47. yawA
    x47 = acclPitch[0:len(x0)]+ np.random.normal(0.05,0.00861,len(x0)) + sinA[0:len(x0)]#; #(1/Iy)*(b*(x[-1+29]+x[-1+28]) - a*(x[-1+27]+x[-1+26]))#;  # 48. pitchA
    x48 = acclRoll[0:len(x0)]+ np.random.normal(0.5,0.00861,len(x0)) + cosA[0:len(x0)]#; #(1/Ix)*(ms*hs*(-x[-1+16]+g*x[-1+7]) - kroll*x[-1+7] - croll*x[-1+8])#; #ms*g*h*x[-1+7] - ms*h*(x[-1+16] + x[-1+14]*x[-1+8]) + (x[-1+26]+x[-1+29]-x[-1+28]-x[-1+27])*d#;       # 49. rollA

    print('I am writing to file in data')
    with open('/home/equaltrace/catkin_ws/src/viso2-indigo/viso2_ros/data/Statedata1.csv', 'wb') as fi:
        write = csv.writer(fi)
        write.writerows(izip(x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30,x31,x32,x33,x34,x35,x36,x37,x38,x39,x40,x41,x42,x43,x44,x45,x46,x47,x48,x49))
    return

def ekf(fstate,x,P,hmeas,z,Q,R):

    x1,A = jaccsd(fstate,x)                     #nonlinear update and linearization at current state
    P = A*P*A.T+Q                               #partial update
    z1,H = jaccsd(hmeas,x1)                     #nonlinear measurement and linearization
    P12=P.dot(H.T)                                   #cross covariance
    # K=P12*inv(H*P12+R);                       #Kalman filter gain
    # x=x1+K*(z-z1);                            #state estimate
    # P=P-K*P12';                               #state covariance matrix
    R=np.linalg.cholesky((H.dot(P12)+R))             #Cholesky factorization
    print(R.shape)
    U=P12.dot(np.linalg.pinv(R))                                   #K=U/R'; Faster because of back substitution
    print(U.shape)
    print(P.shape)
    Temp1 = np.array(z)-np.array(z1)
    Temp2 = np.linalg.solve(R.T,Temp1)
    x=x1+U.dot(Temp2)        #Back substitution to get state update
    P=P-U.dot(U.T)                                   #Covariance update, U*U'=P12/R/R'*P12'=K*P12.

    return x, P


# Code starts

q_proc = 0.1
r_meas = 0.1

Q_proc = (q_proc**2)*np.identity(n_states)
Q_proc[4,4] = 0.01**2
Q_proc[5,5] = 0.01**2
Q_proc[6,6] = 0.01**2
Q_proc[7,7] = 0.01**2
Q_proc[8,8] = 0.01**2
Q_proc[9,9] = 0.01**2
Q_proc[46,46] = 0.01**2
Q_proc[47,47] = 0.01**2
Q_proc[48,48] = 0.01**2

Q_proc_e, Q_proc_E = np.linalg.eig(Q_proc)  # value vector

R_meas = (r_meas**2)*np.identity(n_meas_states)
R_meas[2,2] = 0.01**2
R_meas[3,3] = 0.01**2
R_meas[4,4] = 0.01**2
R_meas[8,8] = 0.01**2
#R_meas[9,9] = 0.01**2
#R_meas[10,10] = 0.01**2

R_meas_e, R_meas_E = np.linalg.eig(R_meas)

#Load data
#datafile = '/home/equaltrace/catkin_ws/src/viso2-indigo/viso2_ros/scripts/03.csv'
datafile = "/home/equaltrace/catkin_ws/src/viso2-indigo/viso2_ros/scripts/03.csv"

row1_e1, row1_e2, row1_e3, row1_e4, row2_e1, row2_e2, row2_e3, row2_e4, row3_e1, row3_e2, row3_e3, row3_e4 = np.loadtxt(datafile, delimiter=',', unpack=True)



iterLen = len(row1_e1)
yaw_meas_S = []
pitch_meas_S = []
roll_meas_S = []
for i in range(0, iterLen):
    inMat = [[row1_e1[i], row1_e2[i], row1_e3[i]], [row2_e1[i], row2_e2[i], row2_e3[i]], [row3_e1[i], row3_e2[i], row3_e3[i]]]
    zYaw, yPitch, xRoll = mat2euler(inMat)
    yaw_meas_S.append(zYaw)
    pitch_meas_S.append(yPitch)
    roll_meas_S.append(xRoll)

S = len(yaw_meas_S)
#NOISE
'''
mu = np.zeros((5,1))
QR = np.diag([0.1**2, 0.1**2, 0.1**2, 0.1**2, 0.1**2])

n = len(QR[:,1])
QRe, QRE = np.linalg.eig(QR)



ra = np.zeros((n,S))
q = np.zeros((n,S))

for i in range(0,S):
    ra[:,i] = np.random.normal(0,1,n)
    q[:,i] = mu + QRE*sqrt(QRe)*ra[:,i]

'''

gt = np.zeros((iterLen, 6))
gt[:,0] =  row3_e4 # x
gt[:,1] = -row1_e4 # y
gt[:,2] = pitch_meas_S # Yaw
gt[:,3] = [el * (-1) for el in roll_meas_S] # Pitch
gt[:,4] = [el * (-1) for el in yaw_meas_S] # Roll
gt[:,5] = -row2_e4 # z


velX = np.zeros(S)
velY = np.zeros(S)
velYaw = np.zeros(S)
velPitch = np.zeros(S)
velRoll = np.zeros(S)

velX[0] = 0.0
velY[0] = 0.0
velYaw[0] = 0.0
velPitch[0] = 0.0
velRoll[0] = 0.0


for i in range(1,S):
    velX[i] = gt[i,1] - gt[i-1,1]
    velY[i] = gt[i, 2] - gt[i - 1, 2]
    velYaw[i] = gt[i, 3] - gt[i - 1, 3]
    velPitch[i] = gt[i, 4] - gt[i - 1, 4]
    velRoll[i] = gt[i, 5] - gt[i - 1, 5]


gt_meas = np.zeros((6,S))
gt_meas[0,:] = gt[:,0] + np.random.normal(0,0.1,S)
gt_meas[1,:] = gt[:,1] + np.random.normal(0,0.1,S)
gt_meas[2,:] = gt[:,2] + np.random.normal(0,0.1,S)
gt_meas[3,:] = gt[:,3] + np.random.normal(0,0.1,S)
gt_meas[4,:] = gt[:,4] + np.random.normal(0,0.1,S)
gt_meas[5,:] = gt[:,5] + np.random.normal(0,0.1,S)



s_state = []

for i in range(0,n_states):
    s_state.append(1.0)

s_state[0] = gt_meas[1,1]
s_state[2] = gt_meas[2,1]
s_state[4] = gt_meas[3,1]
s_state[6] = gt_meas[5,1]
s_state[8] = gt_meas[4,1]

x = s_state + q_proc*np.random.normal(0,0.1,n_states)

P = 100*np.identity(n_states)

# Preallocation for Storage
x0  = [] # x
x1  = [] # x vel
x2  = [] # z
x3  = [] # z vel
x4  = [] # yaw
x5  = [] # yaw rate
x6  = [] # roll
x7  = [] # roll rate
x8  = [] # pitch
x9  = [] # pitch rate
x10 = [] # y
x11 = [] # y vel
x12 = [] # Vxy
x13 = [] #ax
x14 = [] # az
x15 = [] # ay
x16 = [] # Axy
x17 = [] # fxFL
x18 = [] # fxFR
x19 = [] # fxRL
x20 = [] # fxRR
x21 = [] # fyFL
x22 = [] # fyFR
x23 = [] # fyRL
x24 = [] # fyRR
x25 = [] # fzFL
x26 = [] # fzFR
x27 = [] # fzRL
x28 = [] # fzRR
x29 = [] # slip ratio FL
x30 = [] # slip ratio FR
x31 = [] # slip ratio RL
x32 = [] # slip ratio RR
x33 = [] # wheel angular velocity FL
x34 = [] # FR
x35 = [] # RL
x36 = [] # RR
x37 = [] # FL acutal wheel velocity
x38 = [] # FR
x39 = [] # RL
x40 = [] # RR
x41 = [] # Beta
x42 = [] # alpha FL
x43 = [] # alpha FR
x44 = [] # alpha RL
x45 = [] # alpha RR
x46 = [] # yawA
x47 = [] # pitchA
x48 = [] #rollA
x49 = [] # delta

e = q_proc*np.random.normal(0,1,n_states)
d = r_meas*np.random.normal(0,1,n_meas_states)
iterI = 0
#p.pose.position.z
#-p.pose.position.x
#-p.pose.position.y
meas = [1,  1, 1,
        gt_meas[2,iterI], gt_meas[3,iterI], gt_meas[4,iterI],
        gt_meas[0,iterI],gt_meas[1,iterI],gt_meas[5,iterI]]

x, P = ekf(x_handle,x,P,y_handle,meas,Q_proc,R_meas)

# Save states for Plotting
x0.append(float(x[0]))  # x
x1.append(float(x[1]))  # x vel
x2.append(float(x[2]))  # z
x3.append(float(x[3]))  # z vel
x4.append(float(x[4]))  # yaw
x5.append(float(x[5]))  # yaw rate
x6.append(float(x[6]))  # roll
x7.append(float(x[7]))  # roll rate
x8.append(float(x[8]))  # pitch
x9.append(float(x[9]))  # pitch rate
x10.append(float(x[10]))  # y
x11.append(float(x[11]))  # y vel
x12.append(float(x[12]))  # Vxy
x13.append(float(x[13]))  # ax
x14.append(float(x[14]))  # az
x15.append(float(x[15]))  # ay
x16.append(float(x[16]))  # Axy
x17.append(float(x[17]))  # fxFL
x18.append(float(x[18]))  # fxFR
x19.append(float(x[19]))  # fxRL
x20.append(float(x[20]))  # fxRR
x21.append(float(x[21]))  # fyFL
x22.append(float(x[22]))  # fyFR
x23.append(float(x[23]))  # fyRL
x24.append(float(x[24]))  # fyRR
x25.append(float(x[25]))  # fzFL
x26.append(float(x[26]))  # fzFR
x27.append(float(x[27]))  # fzRL
x28.append(float(x[28]))  # fzRR
x29.append(float(x[29]))  # slip ratio FL
x30.append(float(x[30]))  # slip ratio FR
x31.append(float(x[31]))  # slip ratio RL
x32.append(float(x[32]))  # slip ratio RR
x33.append(float(x[33]))  # wheel angular velocity FL
x34.append(float(x[34]))  # FR
x35.append(float(x[35]))  # RL
x36.append(float(x[36]))  # RR
x37.append(float(x[37]))  # FL acutal wheel velocity
x38.append(float(x[38]))  # FR
x39.append(float(x[39]))  # RL
x40.append(float(x[40]))  # RR
x41.append(float(x[41]))  # Beta
x42.append(float(x[42]))  # alpha FL
x43.append(float(x[43]))  # alpha FR
x44.append(float(x[44]))  # alpha RL
x45.append(float(x[45]))  # alpha RR
x46.append(float(x[46]))  # yawA
x47.append(float(x[47]))  # pitchA
x48.append(float(x[48]))  # rollA
x49.append(float(x[49]))  # delta
'''
with open('StateData.csv', 'wb') as fi:
    write = csv.writer(fi)
    write.writerows(izip(x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25,x26,x27,x28,x29,x30,x31,x32,x33,x34,x35,x36,x37,x38,x39,x40,x41,x42,x43,x44,x45,x46,x47,x48,x49))

rospy.loginfo(rospy.get_caller_id() + '  I finished %s', iterI-1)
'''
