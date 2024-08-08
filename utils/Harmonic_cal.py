import numpy as np
from sympy import pi, sin, symbols, integrate, simplify, factor

"""
计算一些关于位移、速度、加速度和Sm的关系表达式
"""

x = symbols('x')
Ts, Tj, Ta, Tv = symbols('Ts Tj Ta Tv')
Jm, Am, Vm, Dm, Sm = symbols('Jm Am Vm Dm Sm')

A1 = Ts * Jm / 2 * (x - 1 / pi * sin(pi * x))
A2 = A1.subs(x, 1) + Tj * Jm * x
A3 = A2.subs(x, 1) + Ts * Jm / 2 * (x + 1 / pi * sin(pi * x))
A4 = A3.subs(x, 1)
A5 = A4.subs(x, 1) + Ts * Jm / 2 * (-x + 1 / pi * sin(pi * x))
A6 = A5.subs(x, 1) - Tj * Jm * x
A7 = A6.subs(x, 1) + Ts * Jm / 2 * (-x - 1 / pi * sin(pi * x))

V1 = Ts * integrate(A1, x)
V2 = V1.subs(x, 1) + Tj * integrate(A2, x)
V3 = V2.subs(x, 1) + Ts * integrate(A3, x)
V4 = V3.subs(x, 1) + Ta * integrate(A4, x)
V5 = V4.subs(x, 1) + Ts * integrate(A5, x)
V6 = V5.subs(x, 1) + Tj * integrate(A6, x)
V7 = V6.subs(x, 1) + Ts * integrate(A7, x)

D1 = Ts * integrate(V1, x)
D2 = D1.subs(x, 1) + Tj * integrate(V2, x)
D3 = D2.subs(x, 1) + Ts * integrate(V3, x)
D4 = D3.subs(x, 1) + Ta * integrate(V4, x)
D5 = D4.subs(x, 1) + Ts * integrate(V5, x)
D6 = D5.subs(x, 1) + Tj * integrate(V6, x)
D7 = D6.subs(x, 1) + Ts * integrate(V7, x)
D8 = D7.subs(x, 1) + Tv * V7.subs(x, 1)

jmax = 2 * Ts * Sm / pi
amax = factor(A3.subs(x, 1))
vmax = factor(V7.subs(x, 1))
dmax = factor(D8.subs(x, 1) + D7.subs(x, 1))
amax = amax.subs(Jm, jmax)
vmax = vmax.subs(Jm, jmax)
dmax = dmax.subs(Jm, jmax)
dmax_1 = dmax.subs({Tj: 0, Ta: 0, Tv: 0})
vmax_1 = vmax.subs({Tj: 0, Ta: 0})
amax_1 = amax.subs({Tj: 0})
dmax_2 = dmax.subs({Ta:0,Tv:0})
vmax_2 = vmax.subs({Ta:0})
amax_2 = amax

print(1)
