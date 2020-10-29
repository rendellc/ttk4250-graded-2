# %% imports
import scipy
import scipy.io
import scipy.stats

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import plotutils as plot

import pickle

try: # see if tqdm is available, otherwise define it as a dummy
    try: # Ipython seem to require different tqdm.. try..except seem to be the easiest way to check
        __IPYTHON__
        from tqdm.notebook import tqdm
    except:
        from tqdm import tqdm
except Exception as e:
    print(e)
    print(
        "install tqdm (conda install tqdm, or pip install tqdm) to get nice progress bars. "
    )

    def tqdm(iterable, *args, **kwargs):
        return iterable

from eskf import (
    ESKF,
    POS_IDX,
    VEL_IDX,
    ATT_IDX,
    ACC_BIAS_IDX,
    GYRO_BIAS_IDX,
    ERR_ATT_IDX,
    ERR_ACC_BIAS_IDX,
    ERR_GYRO_BIAS_IDX,
)

from quaternion import quaternion_to_euler
from cat_slice import CatSlice

# %% plot config check and style setup


# to see your plot config
print(f"matplotlib backend: {matplotlib.get_backend()}")
print(f"matplotlib config file: {matplotlib.matplotlib_fname()}")
print(f"matplotlib config dir: {matplotlib.get_configdir()}")
plt.close("all")

# try to set separate window ploting
if "inline" in matplotlib.get_backend():
    print("Plotting is set to inline at the moment:", end=" ")

    if "ipykernel" in matplotlib.get_backend():
        print("backend is ipykernel (IPython?)")
        print("Trying to set backend to separate window:", end=" ")
        import IPython

        IPython.get_ipython().run_line_magic("matplotlib", "")
    else:
        print("unknown inline backend")

print("continuing with this plotting backend", end="\n\n\n")


# set styles
try:
    # installed with "pip install SciencePLots" (https://github.com/garrettj403/SciencePlots.git)
    # gives quite nice plots
    plt_styles = ["science", "grid", "bright", "no-latex"]
    plt.style.use(plt_styles)
    print(f"pyplot using style set {plt_styles}")
except Exception as e:
    print(e)
    print("setting grid and only grid and legend manually")
    plt.rcParams.update(
        {
            # setgrid
            "axes.grid": True,
            "grid.linestyle": ":",
            "grid.color": "k",
            "grid.alpha": 0.5,
            "grid.linewidth": 0.5,
            # Legend
            "legend.frameon": True,
            "legend.framealpha": 1.0,
            "legend.fancybox": True,
            "legend.numpoints": 1,
        }
    )


# %% load data and plot
filename_to_load = "task_real.mat"
loaded_data = scipy.io.loadmat(filename_to_load)

do_corrections = True # TODO: set to false for the last task
if do_corrections:
    S_a = loaded_data['S_a']
    S_g = loaded_data['S_g']
else:
    # Only accounts for basic mounting directions
    S_a = S_g = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

lever_arm = loaded_data["leverarm"].ravel()
timeGNSS = loaded_data["timeGNSS"].ravel()
timeIMU = loaded_data["timeIMU"].ravel()
z_acceleration = loaded_data["zAcc"].T
z_GNSS = loaded_data["zGNSS"].T
z_gyroscope = loaded_data["zGyro"].T
accuracy_GNSS = loaded_data['GNSSaccuracy'].ravel()

Ts_IMU = [0, *np.diff(timeIMU)]

dt = np.mean(np.diff(timeIMU))
steps = len(z_acceleration)
gnss_steps = len(z_GNSS)

# %% Measurement noise
# Continous noise
cont_gyro_noise_std = 4.36e-5 # TODO
cont_acc_noise_std = 1.167e-3 # TODO

# Discrete sample noise at simulation rate used
rate_std_factor = 0.5
rate_std = rate_std_factor * cont_gyro_noise_std*np.sqrt(1/dt)
acc_std_factor = 0.5
acc_std  = acc_std_factor * cont_acc_noise_std*np.sqrt(1/dt)

# Bias values
rate_bias_driving_noise_std = 5e-5# TODO
cont_rate_bias_factor = 0.5
cont_rate_bias_driving_noise_std = cont_rate_bias_factor * rate_bias_driving_noise_std/np.sqrt(1/dt)

acc_bias_driving_noise_std = 4e-3# TODO
cont_acc_bias_factor = 3
cont_acc_bias_driving_noise_std = cont_acc_bias_factor * acc_bias_driving_noise_std/np.sqrt(1/dt)

# Position and velocity measurement
p_acc = 1e-16 # TODO

p_gyro = 1e-16 # TODO

# %% Estimator
eskf = ESKF(
    acc_std,
    rate_std,
    cont_acc_bias_driving_noise_std,
    cont_rate_bias_driving_noise_std,
    p_acc,
    p_gyro,
    S_a = S_a, # set the accelerometer correction matrix
    S_g = S_g, # set the gyro correction matrix,
    debug=False # False to avoid expensive debug checks
)


# %% Allocate
x_est = np.zeros((steps, 16))
P_est = np.zeros((steps, 15, 15))

x_pred = np.zeros((steps, 16))
P_pred = np.zeros((steps, 15, 15))

NIS = np.zeros(gnss_steps)
NIS_x = NIS.copy()
NIS_y = NIS.copy()
NIS_z = NIS.copy()
NIS_xy = NIS.copy()

# %% Initialise
x_pred[0, POS_IDX] = np.array([0, 0, 0]) # starting 5 metres above ground
x_pred[0, VEL_IDX] = np.array([0, 0, 0]) # starting at 20 m/s due north
x_pred[0, ATT_IDX] = np.array([
    np.cos(45 * np.pi / 180),
    0, 0,
    np.sin(45 * np.pi / 180)
])  # nose to east, right to south and belly down.

P_pred[0][POS_IDX**2] = 7.5**2 * np.eye(3)
P_pred[0][VEL_IDX**2] = 7.5**2 * np.eye(3)
P_pred[0][ERR_ATT_IDX**2] = np.diag([np.pi/30, np.pi/30, np.pi/3])**2
P_pred[0][ERR_ACC_BIAS_IDX**2] = 0.05**2 * np.eye(3)
P_pred[0][ERR_GYRO_BIAS_IDX**2] = (0.005)**2 * np.eye(3)

# %% Run estimation

start = 0
N = 10000

startGNSS = int(start*dt)

timeGNSS = timeGNSS[startGNSS:]
timeIMU = timeIMU[start:]
z_acceleration = z_acceleration[start:]
z_GNSS = z_GNSS[startGNSS:]
z_gyroscope = z_gyroscope[start:]
accuracy_GNSS = accuracy_GNSS[startGNSS:]
Ts_IMU = Ts_IMU[start:]

GNSSk = 0

for k in tqdm(range(N-start)):
    if timeIMU[k] >= timeGNSS[GNSSk]:
        R_GNSS = (0.2*accuracy_GNSS[GNSSk])**2 * np.diag([0.5,0.5,3]) # Current GNSS covariance

        (
            NIS[GNSSk], 
            NIS_x[GNSSk],
            NIS_y[GNSSk],
            NIS_z[GNSSk],
            NIS_xy[GNSSk],
        ) = eskf.NIS_GNSS_position(x_pred[k], P_pred[k], z_GNSS[GNSSk], R_GNSS, lever_arm)

        x_est[k], P_est[k] = eskf.update_GNSS_position(x_pred[k], P_pred[k], z_GNSS[GNSSk], R_GNSS, lever_arm)
        if eskf.debug:
            assert np.all(np.isfinite(P_est[k])), f"Not finite P_pred at index {k}"

        GNSSk += 1
    else:
        # no updates, so estimate = prediction
        x_est[k] = x_pred[k]
        P_est[k] = P_pred[k]

    if k < N - 1:
        x_pred[k + 1], P_pred[k + 1] = eskf.predict(x_est[k], P_est[k], z_acceleration[k+1], z_gyroscope[k+1], Ts_IMU[k+1])

    if eskf.debug:
        assert np.all(np.isfinite(P_pred[k])), f"Not finite P_pred at index {k + 1}"


# %% Plots

figdir = "figs/real/"

fig1, ax1 = plot.trajectoryPlot3D(x_est, [], z_GNSS, N, GNSSk)
fig1.tight_layout()
fig1.savefig(figdir+"ned.pdf")

# state estimation

t = np.linspace(0, dt*(N-1), N)
eul = np.apply_along_axis(quaternion_to_euler, 1, x_est[:N, ATT_IDX])


fig2all, fig2pos, fig2vel, fig2ang, fig2ab, fig2gb = \
        plot.stateplot(t, x_est, eul, N, "States estimates")
fig2all.savefig(figdir+"state_estimates.pdf")

# %% Consistency
confprob = 0.95
CI1 = np.array(scipy.stats.chi2.interval(confprob, 1)).reshape((2, 1))
CI2 = np.array(scipy.stats.chi2.interval(confprob, 2)).reshape((2, 1))
CI3N = np.array(scipy.stats.chi2.interval(confprob, 3 * N)) / N
CI3 = np.array(scipy.stats.chi2.interval(confprob, 3)).reshape((2, 1))
insideCI = np.mean((CI3[0] <= NIS[:GNSSk]) * (NIS[:GNSSk] <= CI3[1]))

fig3, ax3 = plt.subplots(1,1)
plot.plot_NIS(NIS, CI3, "NIS", confprob, dt, N, GNSSk, ax=ax3)
#plt.plot(NIS[:GNSSk])
#plt.plot(np.array([0, N-1]) * dt, (CI3@np.ones((1, 2))).T)
#plt.title(f'NIS ({100 *  insideCI:.1f} inside {100 * confprob} confidence interval)')
#plt.grid()

fig3.savefig(figdir+"nis.pdf")

# %% box plots
fig4, ax4 = plt.subplots(1,1)
plot.boxplot(ax4, [NIS[:GNSSk]], 3, ["NIS"])
fig4.savefig(figdir+"boxplot.pdf")

# %%

# plot NISes
plot.plot_NIS(NIS_x, CI1, "NIS x", confprob, dt, N, GNSSk)
plot.plot_NIS(NIS_y, CI1, "NIS y", confprob, dt, N, GNSSk)
plot.plot_NIS(NIS_z, CI1, "NIS z", confprob, dt, N, GNSSk)
plot.plot_NIS(NIS_xy, CI2, "NIS xy", confprob, dt, N, GNSSk)

ANIS = NIS[:GNSSk].mean()

print(rf'{"ANIS:":<20} {ANIS:^25} {CI3N}')

parameter_changes_from_sim = dict(
    cont_gyro_noise_std = cont_gyro_noise_std, # this should disappear with the code below
    rate_std_factor = rate_std_factor, 
    acc_std_factor = acc_std_factor,
    cont_rate_bias_factor = cont_rate_bias_factor,
    cont_acc_bias_factor = cont_acc_bias_factor,
)


import latexutils
from run_INS_simulated import parameters as sim_parameters
parameters = {**sim_parameters, **parameter_changes_from_sim}
parameter_texvalues = latexutils.parameter_to_texvalues(parameters)
simparameter_texvalues = latexutils.parameter_to_texvalues(sim_parameters)

# filter out the parameters that are the same in sim parameters
remove_keys = []
for k,texvalue in parameter_texvalues.items():
    if texvalue == simparameter_texvalues[k]:
        #print("dbg: remove", k)
        remove_keys.append(k)
for k in remove_keys:
    del parameter_texvalues[k]

latexutils.save_params_to_csv(parameter_texvalues, "csvs/real_params.csv")

# pickle figures so we can review them later
# savedir = "results/real_pickle/"
# 
# def pickle_fig(fig, filename):
#     with open(savedir + filename, 'wb') as f:
#         pickle.dump(fig, f)
# 
# pickle_fig(fig1, "ned.pickle")
# pickle_fig(fig2all, "state_estimate.pickle")
# pickle_fig(fig3, "nis.pickle")
# pickle_fig(fig4, "boxplot.pickle")
# 
