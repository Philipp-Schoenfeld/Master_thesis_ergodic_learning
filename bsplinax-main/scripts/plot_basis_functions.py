import jax.numpy as jnp
import matplotlib.pyplot as plt

from bsplinax.bspline import BsplineBasis, BsplineBasisClamped

if __name__ == "__main__":
    for bspline_class in [BsplineBasis, BsplineBasisClamped]:
        for degree, num_control_points in [(0, 5), (1, 5), (2, 5), (3, 5), (4, 8), (5, 10)]:
            fig, axs = plt.subplots(1, 4, figsize=(16, 4))
            bspline = bspline_class(degree=degree, num_control_points=num_control_points, num_phase_points=248)
            ss = jnp.linspace(0.0, 1.0, bspline.num_phase_points)
            for j, (ax, B) in enumerate(
                zip(axs.flatten(), [bspline.B, bspline.dB_ds, bspline.d2B_ds2, bspline.d3B_ds3], strict=False)
            ):
                # plot basis functions
                for i in range(B.shape[-1]):
                    ax.plot(ss, B[:, i])
                # plot knots
                ax.plot(bspline.knots, jnp.zeros_like(bspline.knots), "x", color="red")
                for knot in bspline.knots:
                    ax.axvline(knot, color="red", linestyle="--", alpha=0.3)
                ax.set_title(f"d^{j}B/ds^{j}", fontsize=10)
                ax.set_xlabel("s")

            fig.suptitle(
                f"B-spline ({bspline.type_bspline}) basis functions"
                f" degree: {bspline.degree},"
                f" num_control_points: {bspline.num_control_points},"
                f" num_knots: {len(bspline.knots)}",
                fontsize=10,
            )
            fig.tight_layout()
    plt.show()
