from quik_fix import plotter
from quik_fix import plot_2d_bar_comparison_from_csv


def test_speedometer_src_text_2():
    plot_2d_bar_comparison_from_csv(
        "sample_data/speedometer-src_text_2-20230308.csv",
        xlabel="Platform",
        ylabel=r"Latency ($\mathrm{s}$)",
    )
    plotter.CONFIG_SAVEFIG_FORMAT = "pdf"
    plot_2d_bar_comparison_from_csv(
        "sample_data/speedometer-src_text_2-20230308.csv",
        xlabel="Platform",
        ylabel=r"Latency ($\mathrm{s}$)",
        annotate_inv_ratio=True,
        plot_avg=True,
    )
