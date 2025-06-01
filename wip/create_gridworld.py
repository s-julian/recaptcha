import numpy as np


def create_gridworld(
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

    # create matrix: dim = n * n; n = matrix_size
    matrix = np.zeros((matrix_size, matrix_size), dtype=int)

    # fill navigable area with 1
    display_height_scaled = int(screenshot_height * scale_factor)
    display_width_scaled = int(screenshot_width * scale_factor)
    matrix[:display_height_scaled, :display_width_scaled] = 1

    # fill target area with 2
    target_x_scaled = int(target_x * scale_factor)  # x_start
    target_y_scaled = int(target_y * scale_factor)  # y_start
    target_w_scaled = int(target_width * scale_factor)
    target_h_scaled = int(target_height * scale_factor)
    y_end = min(target_y_scaled + target_h_scaled, matrix_size)
    x_end = min(target_x_scaled + target_w_scaled, matrix_size)
    matrix[target_y_scaled:y_end, target_x_scaled:x_end] = 2

    # fill mouse location with 3
    mouse_x_scaled = int(mouse_x * scale_factor)
    mouse_y_scaled = int(mouse_y * scale_factor)
    if (
        0 <= mouse_y_scaled < matrix_size
        and 0 <= mouse_x_scaled < matrix_size
    ):
        matrix[mouse_y_scaled][mouse_x_scaled] = 3

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
    import datetime as dt
    import os

    file_dir = os.getcwd()
    data_dir = "data"
    test_dir = "test"
    datetime = dt.datetime.now().strftime("%Y-%m-%d-%H:%M:%S.%f")
    filename = f"gridworld_matrix_{datetime}.csv"
    file_path = os.path.join(file_dir, data_dir, test_dir, filename)

    g1 = create_gridworld(
        screenshot_dim=(3024, 1964),
        target_dim=(264, 62),
        target_loc=(90, 954),
        mouse_loc=(2206, 916),
        matrix_size=1000,
    )

    np.savetxt(file_path, g1, delimiter=",", fmt="%d")
