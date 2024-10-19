import batman
import numpy as np
import matplotlib
from pathlib import Path
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import subprocess

path_default = subprocess.check_output(
    'find "$(pwd -P)" -name ".unique_file_to_find_TizianaWroteOurCodes.txt"',
    shell=True
).decode('utf-8').rstrip('\n').replace(".unique_file_to_find_TizianaWroteOurCodes.txt", '')

planet_name = "WASP107-b"
params = batman.TransitParams()       #object to store transit parameters
params.t0 = 0.                        #time of inferior conjunction
params.per = 5.72148926               #orbital period (in days)
params.rp = 0.1446                    #planet radius (in units of stellar radii)
params.a = 18.2                       #semi-major axis (in units of stellar radii)
params.inc = 89.7                     #orbital inclination (in degrees)
params.ecc = 0.                       #eccentricity
params.w = 90.                        #longitude of periastron (in degrees)
params.limb_dark = "nonlinear"        #limb darkening model
params.u = [0.5, 0.1, 0.1, -0.1]      #limb darkening coefficients [u1, u2, u3, u4]

t14h = 2.753  # transit time between point 1 and 4 in hours
t14d = t14h / 24  # transit time in days
limit_transit = t14d / 2 + 0.3 * t14d 

t = np.linspace(-limit_transit, limit_transit, 5000)  #times at which to calculate light curve (days)
m = batman.TransitModel(params, t)    #initializes model

flux = m.light_curve(params)          #calculates light curve

# Plot the figure
plt.plot(t, flux)
plt.xlabel("Time from central transit")
plt.ylabel("Relative flux")
plt.savefig(str(Path(path_default, "src", "daneel", "detection", "WASP107-b_assignment1_taskF.png")))
plt.savefig(str(Path(path_default, "Results", "assignment1", "WASP107-b_assignment1_taskF.png")))
plt.show()

