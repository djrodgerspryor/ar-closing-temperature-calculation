from plotting import plot_1d, plot_2d
from math import log
import numpy as np
import scipy

DOMAIN_SIZE_POINT_ESTIMATE = 180 # µM
COOLING_RATE_POINT_ESTIMATE = 70 # °C/Ma

DOMAIN_SIZE_RANGE = 120, 250 # µM
COOLING_RATE_RANGE = 35, 150 # (°C difference)/Ma


# Convert degrees Kelvin to degrees Centigrade
def K2C(k):
    return k - 273.15

# Iterative formula for Tc
E_on_R = 32206.1
def Tc_next(Tc_prev, a, dT_on_dt_at_Tc):
    return E_on_R / log((1.077 * 1e20) * Tc_prev**2 / (dT_on_dt_at_Tc * a**2))

# Arguments
#   a             : diffusion domain size in µM
#   dT_on_dt_at_Tc   : cooling rate in °C/Ma
#
# Returns closing temperature in kelvin
def Tc(a, dT_on_dt_at_Tc, Tc_init=450, iterations=10):
    Tc_current = Tc_init

    for i in range(iterations):
        Tc_current = Tc_next(Tc_current, a, dT_on_dt_at_Tc)

    return Tc_current



### Point Estimate ###

print(
    "Point estimate: {:.1f}°C".format(
        K2C(Tc(
            a=DOMAIN_SIZE_POINT_ESTIMATE,
            dT_on_dt_at_Tc=COOLING_RATE_POINT_ESTIMATE
        ))
    )
)



### Variation in domain-size around point estimate ###

a_values = range(*DOMAIN_SIZE_RANGE)
Tc_as_a_function_of_a = []
for a in a_values:
    Tc_as_a_function_of_a.append(
        K2C(Tc(a, COOLING_RATE_POINT_ESTIMATE))
    )

plot_1d(
    x=a_values,
    y=Tc_as_a_function_of_a,
    x_label="a (µM)",
    y_label="Tc (°C)",
    title="Closure Temperature vs a (with dT/dt at Tc = {:.0f}°C/Ma held constant)".format(COOLING_RATE_POINT_ESTIMATE),
    block=False
)



### Variation in cooling-rate around point estimate ###

cooling_rate_values = range(35, 150)
Tc_as_a_function_of_cooling_rate = []
for cooling_rate in cooling_rate_values:
    Tc_as_a_function_of_cooling_rate.append(
        K2C(Tc(DOMAIN_SIZE_POINT_ESTIMATE, cooling_rate))
    )

plot_1d(
    x=cooling_rate_values,
    y=Tc_as_a_function_of_cooling_rate,
    x_label="dT/dt at Tc (°C/Ma)",
    y_label="Tc (°C)",
    title="Closure Temperature vs dT/dt at Tc (with a={:.0f}µM held constant)".format(DOMAIN_SIZE_POINT_ESTIMATE),
    block=False
)



### Multivariate changes around point estimate ###

RESOLUTION = 200

Tc_vec = np.vectorize(Tc)

a_min, a_max = DOMAIN_SIZE_RANGE
cooling_rate_min, cooling_rate_max = COOLING_RATE_RANGE

Tc_matrix = []
for a in np.linspace(a_min, a_max, RESOLUTION):
    Tc_matrix.append([])

    for cooling_rate in np.linspace(cooling_rate_min, cooling_rate_max, RESOLUTION):
        Tc_matrix[-1].append(
            K2C(Tc(a, cooling_rate))
        )
Tc_matrix = np.array(Tc_matrix)

plot_2d(
    Tc_matrix,
    x_label="a (µM)",
    y_label="dT/dt at Tc (°C/Ma)",
    z_label="Tc (°C)",
    title="Closure Temperature Variation",
    extent=[a_min, a_max, cooling_rate_min, cooling_rate_max],
    block=True
)