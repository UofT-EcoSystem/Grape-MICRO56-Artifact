import copy
import csv
import json

import matplotlib.pyplot as plt
import numpy as np

from .logger import logger

CONFIG_SAVEFIG_FORMAT = "png"
CONFIG_USE_HATCH = False
CONFIG_USE_BOTH_HATCH_AND_COLOR = False
_RC_INTIAILIZED = False
_COLOR_CYCLE = plt.rcParams["axes.prop_cycle"].by_key()["color"]
_HATCH_CYCLE = ["", "//", "++", "\\\\", "xx", "||", "--", "__"]


def rc_init(plot_func):
    def _plot_func(*args, **kwargs):
        global _RC_INTIAILIZED
        if not _RC_INTIAILIZED:
            _RC_INTIAILIZED = True
            logger.info("Initializing the RC parameters")
            plt.rc("figure", dpi=1200)
            plt.rc("axes", axisbelow=True)
            plt.rc("text", usetex=True)
            plt.rc(
                "text.latex",
                preamble=r"""
\usepackage{microtype}
\usepackage{amsmath}
\usepackage{nicefrac}
\usepackage{lmodern}
\usepackage{pifont}
\renewcommand{\rmdefault}{ptm}
\renewcommand{\sfdefault}{phv}
""",
            )
            plt.rc("mathtext", rm="serif")
            plt.rc("font", family="serif", size=24)
            logger.info(f"CONFIG_SAVEFIG_FORMAT={CONFIG_SAVEFIG_FORMAT}")
        return plot_func(*args, **kwargs)

    return _plot_func


def _get_color(i):
    if CONFIG_USE_BOTH_HATCH_AND_COLOR:
        return {
            "hatch": _HATCH_CYCLE[i % len(_HATCH_CYCLE)],
            "color": _COLOR_CYCLE[i % len(_COLOR_CYCLE)],
        }
    if CONFIG_USE_HATCH:
        return {"hatch": _HATCH_CYCLE[i % len(_HATCH_CYCLE)], "color": "white"}
    return {"color": _COLOR_CYCLE[i % len(_COLOR_CYCLE)]}


def _transpose_handles(handles, alphas=None, ncol=5):
    """
    Transpose the handles so that they are ordered horizontally rather than
    vertically.

    Parameters
    ----------
    handles
        List of handles to plot
    alphas, optional
        Transparency , by default None
    ncol, optional
        Number of columns, by default 5
    """
    logger.info("Transposing the handles so that they are ordered horizontally")

    num_handles = len(handles)
    num_rows = (num_handles + ncol - 1) // ncol
    num_full_cols = num_handles % ncol

    transposed_handles, transposed_alphas = [], []

    for handle_id, _ in enumerate(handles):
        if handle_id < num_rows * num_full_cols:
            orig_row_id = handle_id % num_rows
            orig_col_id = handle_id // num_rows
        else:
            orig_row_id = (handle_id - num_rows * num_full_cols) % (num_rows - 1)
            orig_col_id = (handle_id - num_rows * num_full_cols) // (
                num_rows - 1
            ) + num_full_cols
        transposed_handles.append(handles[orig_row_id * ncol + orig_col_id])
        if alphas is not None:
            transposed_alphas.append(alphas[orig_row_id * ncol + orig_col_id])

    return transposed_handles, (transposed_alphas if alphas is not None else None)


def save_legend(handles, fig_name, ncol=5, alphas=None, transpose=True):
    """
    Save the legends as a standalone figure.

    Parameters
    ----------
    handles
        List of handles to plot
    fig_name
        Filename of the saved figure
    ncol, optional
        Number of columns, by default 5
    alphas, optional
        Transparency corresponding to each legend handle, by default None
    transpose, optional
        Whether to transpose the legends so that they are ordered horizontally
        rather than vertically, by default True
    """
    lgd_fig = plt.figure()
    plt.axis("off")
    if transpose:
        handles, alphas = _transpose_handles(handles, alphas, ncol)
    lgd = plt.legend(handles=handles, loc="center", ncol=ncol)

    if alphas is not None:
        for i, text in enumerate(lgd.get_texts()):
            text.set_alpha(alphas[i])

    lgd_fig.canvas.draw()
    plt.savefig(
        fig_name,
        bbox_inches=lgd.get_window_extent().transformed(
            lgd_fig.dpi_scale_trans.inverted()
        ),
    )


@rc_init
def plot_2d_bar_comparison(
    data,
    xticklabels,
    labels,
    xlabel,
    ylabel,
    fig_name,
    *,
    bar_width=0.3,
    baseline_idx=0,
    ynbins=None,
    ytop=None,
    figsize=None,
    save_legend_as_fig=False,
    no_legend=False,
    legend_ncol=5,
    annotations_sep_ratio=1.0,
    annotate_inv_ratio=False,
    alt_labels=None,
    plot_avg=False,
    normalize=False,
    inv_normalize=False,
    alphas=None,
    transpose_legends=True,
):
    """
    Plot the 2D data using bar charts.

    Parameters
    ----------
    data
        The 2D data to visualize, of dimension [xticklabels][labels]
    xticklabels
        List of tick labels on the x-axis
    labels
        List of labels on each tick
    xlabel
        Label on the x-axis
    ylabel
        Label on the y-axis
    fig_name
        Name of the saved figure
    bar_width, optional
        Bar width, by default 0.3
    baseline_idx, optional
        Baseline index, by default 0
    ynbins, optional
        Number of bins on the y-axis, by default None
    ytop, optional
        Top boundary of the y-axis, by default None
    figsize, optional
        Figure size, by default None
    save_legend_as_fig, optional
        Whether the legends should be saved as a standalone figure, by default
        False
    no_legend, optional
        Whether the legend should not be plotted, by default False
    legend_ncol, optional
        Number of columns of the legend, by default 5
    annotations_sep_ratio, optional
        Separation ratio of the annotations, by default 1.0
    annotate_inv_ratio, optional
        Whether to annotate the inverse ratios, by default False
    alt_labels, optional
        Alternative labels, by default None
    plot_avg, optional
        Whether to have a separate bar group that represents the average, by
        default False
    normalize, optional
        Whether to normalize all data by the baseline, by default False
    inv_normalize, optional
        Whether to inversely normalize all data by the baseline, by default
        False
    alphas, optional
        Whether to transparentize bars of certain labels, by default None
    transpose_figrues, optional
        Whether to transpose the legends so that they are ordered horizontally
        rather than vertically, by default True
    """
    if figsize is None:
        figsize = (
            (len(labels) * bar_width + 1)
            * (len(xticklabels) + (1 if plot_avg else 0))
            * 1.2,
            6,
        )
        logger.info(f"figsize={figsize}")

    plt.figure(figsize=figsize)

    x_pos = 0
    xticks = [[], []]
    legend_handles = []

    if ynbins is not None:
        logger.info(f"Setting bins={ynbins} along the y-axis")
        plt.locator_params(nbins=ynbins, axis="y")
    else:
        plt.locator_params(nbins=10, axis="y")

    if alt_labels is None:
        alt_labels = labels

    data_avg = {}
    for label in labels:
        data_avg[label] = 0.0

    assert not (
        inv_normalize and (annotate_inv_ratio or normalize)
    ), "The two options cannot be enabled simultaneously"

    data_copy = copy.deepcopy(data)

    for xtick in xticklabels:
        for label in labels:
            if normalize:
                data[xtick][label] = (
                    None
                    if data_copy[xtick][label] is None
                    else (
                        data_copy[xtick][label] / data_copy[xtick][labels[baseline_idx]]
                    )
                )
            elif inv_normalize:
                data[xtick][label] = (
                    None
                    if data_copy[xtick][label] is None
                    else (
                        data_copy[xtick][labels[baseline_idx]] / data_copy[xtick][label]
                    )
                )

    for i, xtick in enumerate(xticklabels):

        def plot_bar(data, xtick):
            nonlocal x_pos  # Avoid the reference before assignment error.
            for j, label in enumerate(labels):
                if data[label] is not None:
                    bar_legend = plt.bar(
                        x_pos,
                        data[label],
                        bar_width,
                        edgecolor="black",
                        linewidth=3,
                        label=alt_labels[j],
                        alpha=alphas[j] if alphas is not None else 1.0,
                        **_get_color(j),
                    )
                    data_avg[label] += data[label] / len(xticklabels)
                else:
                    plt.text(
                        x_pos,
                        0,
                        s=r"\ding{55} OOM",
                        ha="center",
                        va="bottom",
                        backgroundcolor="white",
                        color="red",
                        rotation=90,
                        fontsize=18,
                    )
                if i == 0:
                    legend_handles.append(bar_legend)
                if j == 0:
                    xticks[0].append(x_pos + (len(labels) - 1) / 2 * bar_width)
                    if len(xtick) == 1:
                        xticks[1].append(*xtick)
                    elif len(xtick) == 0:
                        xticks[1].append("")
                    else:
                        xticks[1].append("(" + (",".join(xtick)) + ")")
                x_pos += bar_width
            return x_pos

        plot_bar(data[xtick], xtick)
        x_pos += 1

    # if normalize or inv_normalize:
    #     plt.axhline(y=1.0, color="black", linestyle=(0, (5, 5)))

    if plot_avg:
        plot_bar(data_avg, ("Average",))

    x_pos = 0

    if ytop is not None:
        plt.ylim(top=ytop)

    for i, xtick in enumerate(xticklabels):

        def annotate_bar(data):
            x_center = xticks[0][i]
            nonlocal x_pos
            for j, label in enumerate(labels):
                if data[label] is not None:
                    x_adj = x_center + (x_pos - x_center) * annotations_sep_ratio
                    ratio_to_write = r"{:.3f}$\times$".format(
                        data[label] / data[labels[baseline_idx]]
                    )
                    if j != baseline_idx and annotate_inv_ratio:
                        ratio_to_write = (
                            r"${:.3f}=\nicefrac{{1}}{{{:.3f}}}\times$".format(
                                data[label] / data[labels[baseline_idx]],
                                data[labels[baseline_idx]] / data[label],
                            )
                        )
                    plt.text(
                        x_adj,
                        0.87 * plt.ylim()[1]
                        if annotate_inv_ratio
                        else 0.92 * plt.ylim()[1],
                        ratio_to_write,
                        ha="center",
                        va="center",
                        backgroundcolor="white",
                        rotation=45,
                        fontsize=18,
                        bbox=dict(boxstyle="round", fc="white", alpha=0.9, pad=0.2),
                        alpha=alphas[j] if alphas is not None else 1.0,
                    )
                x_pos += bar_width

        annotate_bar(data[xtick])
        x_pos += 1
    i += 1

    if plot_avg:
        annotate_bar(data_avg)

    plt.xticks(xticks[0], xticks[1])
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.grid(linestyle="-.", linewidth=1)

    assert not (save_legend_as_fig and no_legend), "Contradictory legend configurations"

    if not no_legend:
        if not save_legend_as_fig:
            plt.legend(handles=legend_handles)

    plt.tight_layout()
    plt.savefig(
        fig_name,
        bbox_inches="tight",
        pad_inches=0.05,
    )

    if not no_legend:
        if save_legend_as_fig:
            save_legend(
                legend_handles,
                fig_name.replace(
                    f".{CONFIG_SAVEFIG_FORMAT}", f"-legend.{CONFIG_SAVEFIG_FORMAT}"
                ),
                ncol=legend_ncol,
                alphas=alphas,
                transpose=transpose_legends,
            )


@rc_init
def plot_2d_bar_comparison_from_csv(csv_filename, **kwargs):
    """
    Plot 2D bar comparison from a CSV file.

    Parameters
    ----------
    csv_filename
        Filename of the CSV file
    """
    with open(csv_filename, "rt") as fin:
        csv_reader = csv.DictReader(fin)

        data = {}
        # Use built-in dictionary to keep the insertion order.
        xticklabels = {}
        labels = {}
        for row in csv_reader:
            name, attrs = row["Name"], json.loads(row["Attrs"])
            attr_tuple = tuple(attrs.values())
            if attr_tuple not in data:
                data[attr_tuple] = {}

            if row["Avg"] == "":
                data[attr_tuple][name] = None
            else:
                data[attr_tuple][name] = float(row["Avg"])
            xticklabels[attr_tuple] = None
            labels[name] = None

        logger.info(
            f"data={data}, xticklabels={xticklabels.keys()}, labels={labels.keys()}"
        )

    if "fig_name" not in kwargs:
        kwargs["fig_name"] = csv_filename.replace(".csv", f".{CONFIG_SAVEFIG_FORMAT}")
    plot_2d_bar_comparison(
        data=data,
        xticklabels=list(xticklabels.keys()),
        labels=list(labels.keys()),
        **kwargs,
    )


@rc_init
def plot_stack_bar(
    data,
    labels,
    xticklabels,
    ylabel,
    fig_name,
    *,
    bar_width=0.3,
    ynbins=None,
    ytop=None,
    figsize=(8, 5),
    save_legend_as_fig=False,
    no_legend=False,
    legend_ncol=5,
    alt_labels=None,
    alphas=None,
    transpose_legends=True,
):
    """
    Plot the 1D data using stack bars.

    Parameters
    ----------
    data
        The 1D data to visualize, of dimension [labels]
    labels
        List of labels
    xlabel
        Label on the x-axis
    ylabel
        Label on the y-axis
    fig_name
        Name of the saved figure
    bar_width, optional
        Bar width, by default 0.3
    ynbins, optional
        Number of bins on the y-axis, by default None
    ytop, optional
        Top boundary of the y-axis, by default None
    figsize, optional
        Figure size, by default (8, 5)
    save_legend_as_fig, optional
        Whether the legends should be saved as a standalone figure, by default
        False
    no_legend, optional
        Whether the legend should be plotted, by default False
    legend_ncol, optional
        Number of columns of the legend, by default 5
    alt_labels, optional
        Alternative labels, by default None
    alphas, optional
        Whether to transparentize bars of certain labels, by default None
    transpose_legends, optional
        Whether to transpose the legends so that they are ordered horizontally
        rather than vertically, by default True
    """
    plt.figure(figsize=figsize)

    x_pos = 0
    legend_handles = []
    xticks = []

    if ynbins is not None:
        plt.locator_params(nbins=ynbins, axis="y")
    else:
        plt.locator_params(nbins=10, axis="y")

    if alt_labels is None:
        alt_labels = labels

    for i, _ in enumerate(data):
        for j, _ in enumerate(labels):
            bar_legend = plt.bar(
                x_pos,
                data[i][j],
                bar_width,
                bottom=np.sum(data[i][j + 1 :]),
                edgecolor="black",
                linewidth=3,
                label=alt_labels[j],
                alpha=alphas[j] if alphas is not None else 1.0,
                **_get_color(j),
            )
            if i == 0:
                legend_handles.append(bar_legend)

        side, prev_side = True, True  # Right
        prev_percentage = None

        for j, _ in enumerate(labels):
            middle_pos = data[i][j] / 2 + np.sum(data[i][j + 1 :])
            # Do not add annotations for small components (< 5%).
            curr_percentage = data[i][j] / plt.ylim()[1]
            if curr_percentage < 0.05:
                logger.warning(f"Skipping data={data[i][j]}")
                continue
            logger.info(curr_percentage)
            # If the previous component is small, then we switch side.
            if (
                prev_percentage is not None
                and prev_percentage < 0.1
                and curr_percentage < 0.1
            ):
                side = not prev_side

            margin = 0.005 * plt.ylim()[1]
            arrow_top = middle_pos + data[i][j] / 2 - margin
            arrow_bottom = middle_pos - data[i][j] / 2 + margin

            logger.info(f"arrow_top={arrow_top}, arrow_bottom={arrow_bottom}")

            for y_pos in [arrow_top, arrow_bottom]:
                plt.annotate(
                    "",
                    xy=(x_pos + 0.51 * bar_width * (1 if side is True else -1), y_pos),
                    xytext=(
                        x_pos + 0.6 * bar_width * (1 if side is True else -1),
                        y_pos,
                    ),
                    ha="left" if side is True else "right",
                    va="center",
                    arrowprops=dict(
                        arrowstyle="-",
                        linewidth=2,
                    ),
                )

            plt.annotate(
                "",
                xy=(x_pos + 0.58 * bar_width * (1 if side is True else -1), arrow_top),
                xytext=(
                    x_pos + 0.58 * bar_width * (1 if side is True else -1),
                    arrow_bottom,
                ),
                ha="left" if side is True else "right",
                va="center",
                arrowprops=dict(
                    arrowstyle="-",
                    linewidth=2,
                ),
            )

            plt.annotate(
                "",
                xy=(x_pos + 0.58 * bar_width * (1 if side is True else -1), middle_pos),
                xytext=(
                    x_pos + 0.7 * bar_width * (1 if side is True else -1),
                    middle_pos,
                ),
                ha="left" if side is True else "right",
                va="center",
                arrowprops=dict(
                    arrowstyle="-",
                    linewidth=2,
                ),
            )

            plt.annotate(
                "$%.0f\\%%$" % (data[i][j] * 100.0 / sum(data[i])),
                xy=(x_pos + 0.6 * bar_width * (1 if side is True else -1), middle_pos),
                xytext=(
                    x_pos + 0.7 * bar_width * (1 if side is True else -1),
                    middle_pos,
                ),
                ha="left" if side is True else "right",
                va="center",
                bbox=dict(boxstyle="square", facecolor="white", linewidth=3),
            )

            prev_side = side
            prev_percentage = curr_percentage
        xticks.append(x_pos)
        x_pos += 1

    if len(data) == 2:
        for i, _ in enumerate(labels):
            if data[1][i:] >= data[0][i:]:
                ratio_to_plot = "$%.2f\\times$" % (data[1][i] / data[0][i])
            else:
                ratio_to_plot = "$\\nicefrac{1}{%.2f\\times}$" % (
                    data[0][i] / data[1][i]
                )
            y_dst = sum(data[1][i:]) - data[1][i] / 2
            y_src = sum(data[0][i:]) - data[0][i] / 2
            plt.text(
                x=0.618,
                y=(y_src * 0.618 + y_dst) / 1.618,
                s=ratio_to_plot,
                ha="center",
                va="center",
                backgroundcolor="white",
                fontsize=24,
            )
            plt.annotate(
                "",
                xy=(1 - 0.5 * bar_width, y_dst),
                xytext=(0.5 * bar_width, y_src),
                ha="center",
                va="center",
                zorder=-1,
                arrowprops=dict(
                    arrowstyle="->",
                    linewidth=2,
                ),
            )

    plt.xlim([-1.618 * bar_width, x_pos - 1 + 1.618 * bar_width])
    if ytop is not None:
        plt.ylim(top=ytop)
    logger.info(xticks)
    logger.info(xticklabels)
    plt.xticks(xticks, xticklabels)
    plt.ylabel(ylabel)

    plt.grid(linestyle="-.", linewidth=1, axis="y")

    assert not (save_legend_as_fig and no_legend), "Contradictory legend configurations"

    if not no_legend:
        if not save_legend_as_fig:
            plt.legend(handles=legend_handles)

    plt.tight_layout()
    plt.savefig(fig_name, bbox_inches="tight", pad_inches=0.05)

    if not no_legend:
        if save_legend_as_fig:
            save_legend(
                legend_handles,
                fig_name.replace(
                    f".{CONFIG_SAVEFIG_FORMAT}", f"-legend.{CONFIG_SAVEFIG_FORMAT}"
                ),
                ncol=legend_ncol,
                alphas=alphas,
                transpose=transpose_legends,
            )


@rc_init
def plot_pct_distrib(
    data,
    fig_name,
    explode=0,
    figsize=(8, 5),
    save_legend_as_fig=False,
    legend_ncol=1,
):
    plt.figure(figsize=figsize)

    if isinstance(data, dict):
        data = data.items()
    labels, pcts = zip(*data)
    explode = [explode for _ in labels]
    legend_handles, texts, _ = plt.pie(
        pcts,
        labels=labels if not save_legend_as_fig else None,
        explode=explode,
        autopct="%.1f%%",
        counterclock=False,
        startangle=90,
        shadow=True,
        textprops={"fontsize": 20},
    )
    [text.set_fontsize(24) for text in texts]  # pylint: disable=expression-not-assigned

    if save_legend_as_fig:
        [  # pylint: disable=expression-not-assigned
            legend_handles[i].set_label(labels[i]) for i in range(len(labels))
        ]
        save_legend(
            legend_handles,
            fig_name.replace(
                f".{CONFIG_SAVEFIG_FORMAT}", f"-legend.{CONFIG_SAVEFIG_FORMAT}"
            ),
            ncol=legend_ncol,
        )

    plt.tight_layout()
    plt.savefig(fig_name, bbox_inches="tight", pad_inches=0)
