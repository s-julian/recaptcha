import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def display_gridworld(
    screenshot_dim: tuple[int, int],
    target_dim: tuple[int, int],
    target_loc: tuple[int, int],
    mouse_loc: tuple[int, int],
    matrix_size: int = 1000,
):
    screenshot_width, screenshot_height = screenshot_dim
    target_width, target_height = target_dim
    target_x, target_y = target_loc
    mouse_x, mouse_y = mouse_loc
    side_len = max(screenshot_dim)
    scale_factor = matrix_size / side_len

    if matrix_size >= 1000:
        grid_spacing = 20
        target_min = 40
        mouse_min = 20
    elif matrix_size >= 100:
        grid_spacing = 2
        target_min = 4
        mouse_min = 2
    else:
        grid_spacing = 1
        target_min = 2
        mouse_min = 1

    # create matrix: dim = n * n; n = matrix_size
    matrix = np.zeros((matrix_size, matrix_size), dtype=int)

    # fill navigable area with 1
    display_height_scaled = int(screenshot_height * scale_factor)
    display_width_scaled = int(screenshot_width * scale_factor)
    matrix[:display_height_scaled, :display_width_scaled] = 1

    # fill target area with 2
    target_x_scaled = int(target_x * scale_factor)  # x_start
    target_y_scaled = int(target_y * scale_factor)  # y_start
    target_w_scaled = max(int(target_width * scale_factor), target_min)
    target_h_scaled = max(int(target_height * scale_factor), target_min)
    y_end = min(target_y_scaled + target_h_scaled, matrix_size)
    x_end = min(target_x_scaled + target_w_scaled, matrix_size)
    matrix[target_y_scaled:y_end, target_x_scaled:x_end] = 2

    # fill mouse location with 3
    mouse_x_scaled = int(mouse_x * scale_factor)
    mouse_y_scaled = int(mouse_y * scale_factor)
    # Compute half size for centering
    half_mouse = mouse_min // 2
    # Compute bounds with clipping to matrix edges
    y_start = max(0, mouse_y_scaled - half_mouse)
    y_end = min(matrix_size, mouse_y_scaled + half_mouse + 1)
    x_start = max(0, mouse_x_scaled - half_mouse)
    x_end = min(matrix_size, mouse_x_scaled + half_mouse + 1)
    # Slice assignment
    matrix[y_start:y_end, x_start:x_end] = 3

    # Create visualization
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["black", "white", "red", "green"]
    cmap = ListedColormap(colors)
    im = ax.imshow(  # noqa: F841
        matrix,
        cmap=cmap,
        vmin=0,
        vmax=3,
        origin="upper",
        interpolation="none",
    )
    for i in range(0, matrix_size, grid_spacing):
        ax.axhline(y=i - 0.5, color="gray", linewidth=0.5, alpha=0.3)
        ax.axvline(x=i - 0.5, color="gray", linewidth=0.5, alpha=0.3)
    ax.set_title(f"Gridworld (n={matrix_size})", fontsize=12)
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor="black", label="Non-navigable"),
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor="white",
            edgecolor="black",
            label="Navigable",
        ),
        plt.Rectangle((0, 0), 1, 1, facecolor="red", label="Target region"),
        plt.Rectangle(
            (0, 0), 1, 1, facecolor="green", label="Current location"
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
    )
    plt.tight_layout()
    import datetime as dt
    import os

    file_dir = os.getcwd()
    data_dir = "data"
    test_dir = "test"
    datetime = dt.datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")
    filename = f"gridworld_plot_{datetime}.png"
    file_path = os.path.join(file_dir, data_dir, test_dir, filename)
    plt.savefig(file_path)
    plt.show()

    # print location in matrix
    print(f"Matrix shape: {matrix.shape}")
    print(f"Mouse at: ({mouse_x_scaled}, {mouse_y_scaled})")
    print(f"Target region: ({target_x_scaled}, {target_y_scaled})")
    print(f"Target size: {target_w_scaled}x{target_h_scaled}")
    print(f"Navigable area: {np.sum(matrix == 1)} pixels")
    print(f"Target area: {np.sum(matrix == 2)} pixels")
    print(f"Mouse pixel: {np.sum(matrix == 3)} pixels")
    print(f"Non-navigable: {np.sum(matrix == 0)} pixels")
    return matrix


if __name__ == "__main__":
    display_gridworld(
        screenshot_dim=(3024, 1964),
        target_dim=(264, 62),
        target_loc=(90, 954),
        mouse_loc=(2206, 916),
        matrix_size=1000,
    )
