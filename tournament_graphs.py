import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np


def draw_heatmap(data, labels_x: list, labels_y: list, save_path: str, x_axis_name: str, y_axis_name: str, fmt: str = ".2f", **kwargs):
    words = save_path.split("/")[-1].split(".")[0].split("_")

    for i in range(len(words)):
        words[i] = words[i].capitalize()

    title = " ".join(words)

    if data is list:
        data = np.array(data, dtype=np.float32)

    fig = plt.figure()
    ax = fig.gca()

    if "reverse" in kwargs and kwargs["reverse"]:
        cp = sb.color_palette("rocket_r", as_cmap=True)
    else:
        cp = sb.color_palette("rocket", as_cmap=True)

    vmax = 1.
    vmin = 0.
    if "vmax" in kwargs and kwargs["vmax"]:
        data_c = np.array(data) + np.identity(len(data)) * 1.5

        vmax = np.max(data_c)
        vmin = np.min(data_c)

    data = np.array(data, dtype=np.float32)

    if data.shape[0] == data.shape[1]:
        for i in range(data.shape[0]):
            data[i, i] = None

    hm = sb.heatmap(data=data, annot=True, fmt=fmt, xticklabels=labels_x, yticklabels=labels_y,
                    vmin=vmin, vmax=vmax, ax=ax, square=False, cmap=cp, annot_kws={"fontsize": 16, "weight": "bold"})

    hm.set_facecolor("dimgray")
    hm.set_xticklabels(hm.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=16)
    hm.set_yticklabels(hm.get_yticklabels(), horizontalalignment='right', fontsize=16)

    plt.title(title, fontsize=20)

    plt.xlabel(x_axis_name, fontsize=18)
    plt.ylabel(y_axis_name, fontsize=18)

    # Resize
    fig = plt.gcf()

    fig_width = 2.0 + len(labels_x) / 1.05
    fig_height = 1.0 + len(labels_y) / 1.15

    fig.set_size_inches(fig_width, fig_height)

    plt.tight_layout()

    save_path = save_path[:-3] + "svg"

    plt.savefig(save_path, dpi=1200)
    plt.close()

    if "save_to_csv" not in kwargs or kwargs["save_to_csv"]:
        save_heatmap(data, labels_x, labels_y, save_path, x_axis_name, y_axis_name, title)


def save_heatmap(data: np.ndarray, labels_x: list, labels_y: list, save_path: str, x_axis_name: str, y_axis_name: str, title: str):
    save_path = save_path[:-3] + "csv"

    with open(save_path, "w") as f:
        f.write(title + ";\n\n")

        f.write(";" + x_axis_name + ";\n" + y_axis_name + ";")

        for j in range(len(labels_x)):
            f.write(str(labels_x[j]) + ";")

        f.write("\n")

        for i in range(data.shape[0]):
            f.write(str(labels_y[i]) + ";")

            for j in range(data.shape[1]):
                if data[i][j] is not None:
                    f.write(str(data[i][j]) + ";")
                else:
                    f.write(";")

            f.write("\n")


def draw_line(data: dict, save_path: str, x_axis_name: str, y_axis_name: str, save_to_csv: bool = True):
    words = save_path.split("/")[-1].split(".")[0].split("_")

    for i in range(len(words)):
        words[i] = words[i].capitalize()

    title = " ".join(words)

    for key in data.keys():
        plt.plot(np.array(list(range(len(data[key])))), np.array(data[key]), label=key)

    plt.title(title, fontsize=20)
    plt.xlabel(x_axis_name, fontsize=18)
    plt.ylabel(y_axis_name, fontsize=18)

    plt.legend()

    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)

    plt.tight_layout()

    save_path = save_path[:-3] + "svg"

    plt.savefig(save_path, dpi=1200)
    plt.close()

    if save_to_csv:
        save_line(data, save_path, x_axis_name, y_axis_name, title)


def save_line(data: dict, save_path: str, x_axis_name: str, y_axis_name: str, title: str):
    save_path = save_path[:-3] + "csv"

    with open(save_path, "w") as f:
        max_length = max([len(values) for values in data.values()])

        f.write(title + ";\n\n")

        f.write(x_axis_name + ";" + y_axis_name + ";\n" + x_axis_name + ";")

        key_list = list(data.keys())

        for key in key_list:
            f.write(str(key) + ";")

        f.write("\n")

        for i in range(max_length):
            f.write(str(i) + ";")

            for key in key_list:
                if len(data[key]) <= i:
                    f.write(";")
                else:
                    f.write(str(data[key][i]) + ";")

            f.write("\n")
