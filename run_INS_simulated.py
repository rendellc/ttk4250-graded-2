# %% imports
import scipy
import scipy.io
import scipy.stats

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import plotutils as plot

import latexutils

try: # see if tqdm is available, otherwise define it as a dummy
    try: # Ipython seem to require different tqdm.. try..except seem to be the easiest way to check
        __IPYTHON__
        import tqdm
    except:
        import tqdm
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
filename_to_load = "task_simulation.mat"
loaded_data = scipy.io.loadmat(filename_to_load)

S_a = loaded_data["S_a"]
S_g = loaded_data["S_g"]
lever_arm = loaded_data["leverarm"].ravel()
timeGNSS = loaded_data["timeGNSS"].ravel()
timeIMU = loaded_data["timeIMU"].ravel()
x_true = loaded_data["xtrue"].T
z_acceleration = loaded_data["zAcc"].T
z_GNSS = loaded_data["zGNSS"].T
z_gyroscope = loaded_data["zGyro"].T

Ts_IMU = [0, *np.diff(timeIMU)]

dt = np.mean(np.diff(timeIMU))
steps = len(z_acceleration)
gnss_steps = len(z_GNSS)


def run_experiment(parameters):
    # read parameters
    cont_gyro_noise_std = parameters["cont_gyro_noise_std"]
    cont_acc_noise_std = parameters["cont_acc_noise_std"]
    rate_std_factor = parameters["rate_std_factor"]
    acc_std_factor = parameters["acc_std_factor"]
    rate_bias_driving_noise_std = parameters["rate_bias_driving_noise_std"]
    cont_rate_bias_factor = parameters["cont_rate_bias_factor"]
    acc_bias_driving_noise_std = parameters["acc_bias_driving_noise_std"]
    cont_acc_bias_factor = parameters["cont_acc_bias_factor"]
    p_acc = parameters["p_acc"]
    p_gyro = parameters["p_gyro"]
    p_std = parameters["p_std"]
    sigma_pos = parameters["sigma_pos"]
    sigma_vel = parameters["sigma_vel"]
    sigma_rollpitch = parameters["sigma_rollpitch"]
    sigma_yaw = parameters["sigma_yaw"]
    sigma_err_acc_bias = parameters["sigma_err_acc_bias"]
    sigma_err_gyro_bias = parameters["sigma_err_gyro_bias"]
    N = parameters["N"]
    doGNSS = parameters["doGNSS"]
    debug = parameters["debug"]
    dosavefigures = parameters["dosavefigures"]
    figdir = parameters["figdir"]
    doshowplot = parameters["doshowplot"]

    # derived parameters
    rate_std = rate_std_factor * cont_gyro_noise_std * np.sqrt(1 / dt) # Hvorfor gange med en halv? (eq. 10.70)
    acc_std = acc_std_factor * cont_acc_noise_std * np.sqrt(1 / dt)
    cont_rate_bias_driving_noise_std = cont_rate_bias_factor * rate_bias_driving_noise_std / np.sqrt(1 / dt)
    cont_acc_bias_driving_noise_std = cont_acc_bias_factor * acc_bias_driving_noise_std / np.sqrt(1 / dt)
    R_GNSS = np.diag(p_std ** 2)

    # %% Estimator
    eskf = ESKF(
        acc_std,
        rate_std,
        cont_acc_bias_driving_noise_std,
        cont_rate_bias_driving_noise_std,
        p_acc,
        p_gyro,
        S_a=S_a, # set the accelerometer correction matrix
        S_g=S_g, # set the gyro correction matrix,
        debug=debug # TODO: False to avoid expensive debug checks, can also be suppressed by calling 'python -O run_INS_simulated.py'
    )

    # %% Allocate
    x_est = np.zeros((steps, 16))
    P_est = np.zeros((steps, 15, 15))

    x_pred = np.zeros((steps, 16))
    P_pred = np.zeros((steps, 15, 15))

    delta_x = np.zeros((steps, 15))

    NIS = np.zeros(gnss_steps)
    NIS_x = np.zeros(gnss_steps)
    NIS_y = np.zeros(gnss_steps)
    NIS_z = np.zeros(gnss_steps)
    NIS_xy = np.zeros(gnss_steps)

    NEES_all = np.zeros(steps)
    NEES_pos = np.zeros(steps)
    NEES_vel = np.zeros(steps)
    NEES_att = np.zeros(steps)
    NEES_accbias = np.zeros(steps)
    NEES_gyrobias = np.zeros(steps)

    # %% Initialise
    x_pred[0, POS_IDX] = np.array([0, 0, -5])  # starting 5 metres above ground
    x_pred[0, VEL_IDX] = np.array([20, 0, 0])  # starting at 20 m/s due north
    x_pred[0, 6] = 1  # no initial rotation: nose to North, right to East, and belly down

    # These have to be set reasonably to get good results
    P_pred[0][POS_IDX ** 2] = sigma_pos**2 * np.eye(3)
    P_pred[0][VEL_IDX ** 2] = sigma_vel**2 * np.eye(3)
    P_pred[0][ERR_ATT_IDX ** 2] = np.diag([sigma_rollpitch, sigma_rollpitch, sigma_yaw])**2 
    P_pred[0][ERR_ACC_BIAS_IDX ** 2] = sigma_err_acc_bias**2 * np.eye(3)
    P_pred[0][ERR_GYRO_BIAS_IDX ** 2] = sigma_err_gyro_bias**2 * np.eye(3)

    # %% Run estimation
    # run this file with 'python -O run_INS_simulated.py' to turn of assertions and get about 8/5 speed increase for longer runs

    GNSSk: int = 0  # keep track of current step in GNSS measurements
    for k in tqdm.trange(N):
        if doGNSS and timeIMU[k] >= timeGNSS[GNSSk]:
            (
                NIS[GNSSk], 
                NIS_x[GNSSk],
                NIS_y[GNSSk],
                NIS_z[GNSSk],
                NIS_xy[GNSSk],
            )  = eskf.NIS_GNSS_position(x_pred[k], P_pred[k], z_GNSS[GNSSk], R_GNSS, lever_arm)

            x_est[k], P_est[k] = eskf.update_GNSS_position(x_pred[k], P_pred[k], z_GNSS[GNSSk], R_GNSS, lever_arm)
            assert np.all(np.isfinite(P_est[k])), f"Not finite P_pred at index {k}"
            
            GNSSk += 1
        else:
            # no updates, so let us take estimate = prediction
            x_est[k] = x_pred[k]
            P_est[k] = P_pred[k]

        delta_x[k] = eskf.delta_x(x_est[k], x_true[k])
        (
            NEES_all[k],
            NEES_pos[k],
            NEES_vel[k],
            NEES_att[k],
            NEES_accbias[k],
            NEES_gyrobias[k],
        ) = eskf.NEESes(x_est[k], P_est[k], x_true[k])

        if k < N - 1:
            x_pred[k + 1], P_pred[k + 1] = eskf.predict(x_est[k], P_est[k], z_acceleration[k+1], z_gyroscope[k+1], Ts_IMU[k+1])

        if eskf.debug:
            assert np.all(np.isfinite(P_pred[k])), f"Not finite P_pred at index {k + 1}"

    # %% Consistency
    confprob = 0.95
    CI15 = np.array(scipy.stats.chi2.interval(confprob, 15)).reshape((2, 1))
    CI3 = np.array(scipy.stats.chi2.interval(confprob, 3)).reshape((2, 1))
    CI1 = np.array(scipy.stats.chi2.interval(confprob, 1)).reshape((2, 1))
    CI2 = np.array(scipy.stats.chi2.interval(confprob, 2)).reshape((2, 1))
    CI3N = np.array(scipy.stats.chi2.interval(confprob, 3 * N)) / N
    CI15N = np.array(scipy.stats.chi2.interval(confprob, 15 * N)) / N

    ANIS = NIS[:GNSSk].mean()
    ANEES = NEES_all[:N].mean()
    ANEES_pos = NEES_pos[:N].mean()
    ANEES_vel = NEES_vel[:N].mean()
    ANEES_att = NEES_att[:N].mean()
    ANEES_accbias = NEES_accbias[:N].mean()
    ANEES_gyrobias = NEES_gyrobias[:N].mean()

    print(rf'{"ANEES:":<20} {ANEES:^25} {CI15N}')
    print(rf'{"ANEESS_pos:":<20} {ANEES_pos:^25} {CI3N}')
    print(rf'{"ANEES_vel:":<20} {ANEES_vel:^25} {CI3N}')
    print(rf'{"ANEES_att:":<20} {ANEES_att:^25} {CI3N}')
    print(rf'{"ANEES_accbias:":<20} {ANEES_accbias:^25} {CI3N}')
    print(rf'{"ANEES_gyrobias:":<20} {ANEES_gyrobias:^25} {CI3N}')
    print(rf'{"ANIS:":<20} {ANIS:^25} {CI3N}')

    eul = np.apply_along_axis(quaternion_to_euler, 1, x_est[:N, ATT_IDX])
    eul_true = np.apply_along_axis(quaternion_to_euler, 1, x_true[:N, ATT_IDX])
    wrap_to_pi = lambda rads: (rads + np.pi) % (2 * np.pi) - np.pi
    eul_error = wrap_to_pi(eul[:N] - eul_true[:N]) * 180 / np.pi
    pos_err = np.linalg.norm(delta_x[:N, POS_IDX], axis=1)
    meas_err = np.linalg.norm(x_true[99:N:100, POS_IDX] - z_GNSS[:GNSSk], axis=1)


    # %% plotting
    dosavefigures = True
    doplothandout = True

    t = np.linspace(0, dt * (N - 1), N)
    if doplothandout:
        # 3d position plot
        fig1, ax1 = plot.trajectoryPlot3D(x_est, x_true, z_GNSS, N, GNSSk)
        fig1.tight_layout()
        if dosavefigures: fig1.savefig(figdir+"ned.pdf")

        # state estimation plot

        fig2all, fig2pos, fig2vel, fig2ang, fig2ab, fig2gb = \
                plot.stateplot(t, x_est, eul, N, "States estimates")
        # fig2vehicle = plot.kinematicplot(t, x_est, eul, N, "Kinematic estimates")
        # fig2bias = plot.biasplot(t, x_est, eul, N, "Bias estimates")
        if dosavefigures: 
            fig2all.savefig(figdir+"state_estimates.pdf")
            fig2pos.savefig(figdir+"estimate_pos.pdf")
            fig2vel.savefig(figdir+"estimate_vel.pdf")
            fig2ang.savefig(figdir+"estimate_eul.pdf")
            fig2ab.savefig(figdir+"estimate_aclbias.pdf")
            fig2gb.savefig(figdir+"estimate_gyrobias.pdf")

        # state error plots
        # fig3.tight_layout()

        fig3all, fig3pos, fig3vel, fig3ang, fig3ab, fig3gb = \
                plot.stateerrorplot(t, delta_x, eul_error, N, "State estimate errors")
        if dosavefigures:
            fig3all.savefig(figdir+"state_estimate_errors.pdf")
            fig3pos.savefig(figdir+"estimate_error_pos.pdf")
            fig3vel.savefig(figdir+"estimate_error_vel.pdf")
            fig3ang.savefig(figdir+"estimate_errors_eul.pdf")
            fig3ab.savefig(figdir+"estimate_errors_aclbias.pdf")
            fig3gb.savefig(figdir+"estimate_errors_gyrobias.pdf")

        # Error distance plot
        fig4, axs4 = plt.subplots(2, 1, num=4, clear=True, sharex=True)

        est_error = np.sqrt(np.mean(np.sum(delta_x[:N, POS_IDX]**2, axis=1)))
        meas_error = np.sqrt(np.mean(np.sum((x_true[99:N:100, POS_IDX] - z_GNSS[:GNSSk])**2, axis=1)))
        axs4[0].plot(t, pos_err, label=f"Est error ({est_error:.3f})")
        axs4[0].plot(np.arange(0, N, 100) * dt, meas_err, label=f"GNSS error ({meas_error:.3f})")
        axs4[0].set(title=r"Position error $[m]$")
        axs4[0].legend(loc='upper right')

        vel_rmse = np.sqrt(np.mean(np.sum(delta_x[:N, VEL_IDX]**2, axis=1)))
        axs4[1].plot(t, np.linalg.norm(delta_x[:N, VEL_IDX], axis=1), label=f"RMSE: {vel_rmse:.3f}")
        axs4[1].set(title=r"Speed error $[m/s]$")
        axs4[1].legend(loc='upper right')
        fig4.tight_layout()

        latexutils.save_value("Estimation error", f"{est_error:.3f}", "csvs/sim_est_error_pos.csv")
        latexutils.save_value("GNSS error", f"{meas_error:.3f}", "csvs/sim_meas_error_pos.csv")

        #fig4.tight_layout()
        if dosavefigures:
            fig4.savefig(figdir+"estimate_vs_measurement_error.pdf")


        fig5, axs5 = plt.subplots(4, 1, num=5, clear=True, sharex=True)
        fig5.subplots_adjust(hspace=0.4) # so ytick dont overlap
        fig5.set_size_inches((3.5, 5))

        insideCItot = np.mean((CI15[0] <= NEES_all[:N]) * (NEES_all[:N] <= CI15[1]))
        plot.pretty_NEESNIS(axs5[0], t, NEES_all[:N], "total", CI15, fillCI=True, upperY=50)
        axs5[0].legend(loc="upper right",ncol=1)


        plot.pretty_NEESNIS(axs5[1], t, NEES_pos[:N], "pos", CI3, fillCI=True, upperY=20)
        plot.pretty_NEESNIS(axs5[1], t, NEES_vel[:N], "vel", CI3, fillCI=False, upperY=20)
        plot.pretty_NEESNIS(axs5[1], t, NEES_att[:N], "att", CI3, fillCI=False, upperY=20)
        #axs5[1].plot([t[0], t[~0]], (CI3 @ np.ones((1, 2))).T)
        # axs5[1].plot(t, (NEES_pos[0:N]).T, label="pos")
        # axs5[1].plot(t, (NEES_vel[0:N]).T, label="vel")
        # axs5[1].plot(t, (NEES_att[0:N]).T, label="att")
        # axs5[1].set_ylim([0, 20])
        axs5[1].legend(loc="best",ncol=3)

        #axs5[2].plot([t[0], t[~0]], (CI3 @ np.ones((1, 2))).T)
        plot.pretty_NEESNIS(axs5[2], t, NEES_accbias[:N], "accel bias", CI3, fillCI=True, upperY=20)
        plot.pretty_NEESNIS(axs5[2], t, NEES_gyrobias[:N], "gyro bias", CI3, fillCI=False, upperY=20)
        # axs5[2].plot(t, (NEES_accbias[0:N]).T,  label="accel bias")
        # axs5[2].plot(t, (NEES_gyrobias[0:N]).T, label="gyro bias")
        # axs5[2].set_ylim([0, 20])
        axs5[2].legend(loc="best",ncol=2)
        
        t_gnssk = np.linspace(t[0], t[~0], GNSSk)
        plot.pretty_NEESNIS(axs5[3], t_gnssk, NIS[:GNSSk], "NIS", CI3, fillCI=True, upperY=20)
        # axs5[3].plot([t[0], t[~0]], (CI3 @ np.ones((1, 2))).T)
        # axs5[3].plot(NIS[:GNSSk], label="NIS")
        # axs5[3].set_ylim([0, 20])
        axs5[3].legend(loc="upper right")

        for ax in axs5:
            ax.set_xlim([t[0], t[~0]])

        #fig5.tight_layout()

        insideCIpos = np.mean((CI3[0] <= NEES_pos[:N]) * (NEES_pos[:N] <= CI3[1]))
        insideCIvel = np.mean((CI3[0] <= NEES_vel[:N]) * (NEES_vel[:N] <= CI3[1]))
        insideCIatt = np.mean((CI3[0] <= NEES_att[:N]) * (NEES_att[:N] <= CI3[1]))
        insideCIab = np.mean((CI3[0] <= NEES_accbias[:N]) * (NEES_accbias[:N] <= CI3[1]))
        insideCIgb = np.mean((CI3[0] <= NEES_gyrobias[:N]) * (NEES_gyrobias[:N] <= CI3[1]))
        insideCInis = np.mean((CI3[0] <= NIS[:GNSSk]) * (NIS[:GNSSk] <= CI3[1]))

        print("Inside CI total", insideCItot)
        print("Inside CI pos", insideCIpos)
        print("Inside CI vel", insideCIvel)
        print("Inside CI att", insideCIatt)
        print("Inside CI ab", insideCIab)
        print("Inside CI gb", insideCIgb)
        print("Inside CI NIS", insideCInis)


        #fig5.tight_layout()
        if dosavefigures: 
            fig5.savefig(figdir+"nees_nis.pdf")

        # boxplot
        fig6, axs6 = plt.subplots(1, 3, 
                gridspec_kw={"width_ratios":[1,1,2]}
        )
        plot.boxplot(axs6[0], [NIS[:GNSSk]], 3, ["NIS"])
        plot.boxplot(axs6[1], [NEES_all[0:N].T], 15, ["NEES"])
        plot.boxplot(axs6[2], 
            [NEES_pos[0:N].T, NEES_vel[0:N].T, NEES_att[0:N].T, NEES_accbias[0:N].T, NEES_gyrobias[0:N].T],
            3,
            ['pos', 'vel', 'att', 'accbias', 'gyrobias'])
        fig6.tight_layout()

        #fig6.tight_layout()
        if dosavefigures: 
            fig6.savefig(figdir+"boxplot.pdf")


        plot.plot_NIS(NIS_x, CI1, "NIS_x", confprob, dt, N, GNSSk)
        plot.plot_NIS(NIS_y, CI1, "NIS_y", confprob, dt, N, GNSSk)
        plot.plot_NIS(NIS_z, CI1, "NIS_z", confprob, dt, N, GNSSk)
        plot.plot_NIS(NIS_xy, CI2, "NIS_xy", confprob, dt, N, GNSSk)

        consistencydatas = [
                dict(avg=ANEES,inside=insideCItot, text="NEES",CI=CI15N),
                dict(avg=ANEES_pos,inside=insideCIpos, text="NEES pos",CI=CI3N),
                dict(avg=ANEES_vel,inside=insideCIvel, text="NEES vel",CI=CI3N),
                dict(avg=ANEES_att,inside=insideCIatt, text="NEES att",CI=CI3N),
                dict(avg=ANEES_gyrobias,inside=insideCIgb, text="NEES gyro bias",CI=CI3N),
                dict(avg=ANEES_accbias,inside=insideCIab, text="NEES acc bias",CI=CI3N),
                dict(avg=ANIS,inside=insideCInis, text="NIS",CI=CI3N),
        ]

        latexutils.save_consistency_results(consistencydatas, "csvs/sim_consistency.csv")



    if doshowplot:
        plt.show()

parameters = dict(
    cont_gyro_noise_std = 4.36e-5,  # (rad/s)/sqrt(Hz)
    cont_acc_noise_std = 1.167e-3,  # (m/s**2)/sqrt(Hz)
    # Discrete sample noise at simulation rate used
    rate_std_factor = 0.3, 
    acc_std_factor = 0.3,
    # Bias values
    rate_bias_driving_noise_std = 5e-5,
    cont_rate_bias_factor = 1,
    acc_bias_driving_noise_std = 4e-3,
    cont_acc_bias_factor = 6,
    p_acc = 1e-16,
    p_gyro = 1e-16,
    # Position and velocity measurement
    # Measurement noise
    p_std = np.array([0.3, 0.3, 0.5]),
    # Initial covariances
    sigma_pos = 7.5,
    sigma_vel = 7.5,
    sigma_rollpitch = np.pi/30,
    sigma_yaw = np.pi/3,
    sigma_err_acc_bias = 0.05,
    sigma_err_gyro_bias = 0.005,
    # Simulation parameters
    dt = dt,
    N = 1000,
    doGNSS = True,
    debug = False,
    dosavefigures = True,
    doshowplot = False,
    figdir="figs/simulated_all/",
    dopickle = False,
)

if __name__ == "__main__":
    run_experiment(parameters)

    parameter_texvalues = latexutils.parameter_to_texvalues(parameters)
    latexutils.save_params_to_csv(parameter_texvalues, "csvs/simulated_params.csv")




