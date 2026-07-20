from matplotlib import pyplot as plt

from bsplinax.phase_time import PhaseTimeLinear, PhaseTimeParabola

if __name__ == "__main__":
    for phase_time_cls in [PhaseTimeLinear, PhaseTimeParabola]:
        phase_time = phase_time_cls(total_duration=3.0, num_time_points=512)

        fig, axs = plt.subplots(figsize=(16, 4), nrows=1, ncols=5, squeeze=False)

        axs[0, 0].plot(phase_time.tt, phase_time.ss, color="orange")
        axs[0, 0].set_title("s = phi^{-1}(t)")
        axs[0, 0].set_xlabel("t")
        axs[0, 0].set_ylabel("s")

        axs[0, 1].plot(phase_time.ss, phase_time.tt, color="orange")
        axs[0, 1].set_title("t = phi(s)")
        axs[0, 1].set_xlabel("s")
        axs[0, 1].set_ylabel("t")

        axs[0, 2].plot(phase_time.ss, phase_time.r, color="orange")
        axs[0, 2].set_title("r(s) = ds/dt(s)")
        axs[0, 2].set_xlabel("s")
        axs[0, 2].set_ylabel("r")

        axs[0, 3].plot(phase_time.ss, phase_time.dr_ds, color="orange")
        axs[0, 3].set_title("dr/ds(s) = d^2s/dt^2(s)")
        axs[0, 3].set_xlabel("s")
        axs[0, 3].set_ylabel("dr/ds")

        axs[0, 4].plot(phase_time.ss, phase_time.d2r_ds2, color="orange")
        axs[0, 4].set_title("d^2r/ds^2(s) = d^3s/dt^3(s)")
        axs[0, 4].set_xlabel("s")
        axs[0, 4].set_ylabel("d^2r/ds^2")

        fig.suptitle(f"Phase-Time mapping: {phase_time_cls.__name__}", fontsize=16)

        fig.tight_layout()

    plt.show()
