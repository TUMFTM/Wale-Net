import sys
import os

# Third party imports
import numpy as np
import cv2
import matplotlib.pyplot as plt

from commonroad.visualization.draw_dispatch_cr import draw_object

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    import mod_prediction  # noqa: F401

    __package__ = "mod_prediction"

# Custom imports
from mod_prediction.utils.geometry import point_in_rectangle, abs_to_rel_coord


def generate_self_rendered_sc_img(
    watch_radius, scenario, curr_pos, curr_orient, res=256, light_lane_dividers=True
):
    """Render scene image in relative position."""
    # region generate_self_rendered_sc_img()
    # region read inputs
    lane_net = scenario.lanelet_network
    pixel_dist = 2 * watch_radius / res
    interp_factor = 0.8
    # endregion
    # timer = time.time()

    # region read all lanelet boundarys into a list
    # IDEA speed up scene rendering even more multiprocessing

    bound_list = []
    type_list = []
    for lanelet in lane_net.lanelets:
        bound_list.append(lanelet.left_vertices)
        line_type = "road-boundary" if lanelet.adj_left is None else "lane-marking"
        type_list.append(line_type)
        bound_list.append(lanelet.right_vertices)
        line_type = "road-boundary" if lanelet.adj_right is None else "lane-marking"
        type_list.append(line_type)
    # endregion

    # region translate rotate image
    bound_list = [
        abs_to_rel_coord(curr_pos, curr_orient, bound_line) for bound_line in bound_list
    ]
    # endregion
    # print(f"Time for reading points:{time.time() - timer}")
    # timer = time.time()

    # region limit_boundarys to watch_radius
    # region limit_boundary_subfunction
    def limit_boundary(boundary):
        array = np.empty(len(boundary))
        last_point_was_out = None  # This line makes the linter happy
        # Loop over all points
        for index, point in enumerate(boundary):
            # Check index to avoid indexerrors
            if index > 0:
                # check if point is outside of viewing window
                point_is_out = bool(max(abs(point)) > watch_radius)
                # If point is inside
                if point_is_out is False:
                    array[index] = False
                    # Add the neighbour, so that a continous line
                    # to the image border can be rendered
                    if last_point_was_out is True:
                        array[index - 1] = False
                # if point is outside of watch_radius
                else:
                    # Add this point as neighbor if the last point was in
                    if last_point_was_out is False:
                        array[index] = False
                    # Remove point from boundary line
                    else:
                        array[index] = True
            else:
                # Handling of first element
                point_is_out = bool(max(abs(point)) > watch_radius)
                array[index] = point_is_out
            last_point_was_out = point_is_out
        return array

    # endregion
    # Call the function
    limit_bound_list = [
        np.delete(bound, limit_boundary(bound).astype(bool), axis=0)
        for bound in bound_list
    ]
    # endregion

    # print(f"Time for limiting array:{time.time() - timer}")
    # timer = time.time()
    # region Interpolate boundary lines

    # region interpolate_boundary() subfunction
    def interpolate_boundary(boundary):
        # region calc curve length of boundary
        curve_length = np.zeros(len(boundary))
        bound_array = np.array(boundary)
        for index, point in enumerate(bound_array[1:], start=1):
            curve_length[index] = curve_length[index - 1] + np.linalg.norm(
                point - boundary[index - 1]
            )
        # endregion
        # region interpolate over curve_length
        if len(curve_length) > 0:
            eval_array = np.arange(0, curve_length[-1], pixel_dist * interp_factor)
            return np.array(
                [
                    np.interp(eval_array, curve_length, bound_array.transpose()[0]),
                    np.interp(eval_array, curve_length, bound_array.transpose()[1]),
                ]
            )
        # if no point is left return None
        return None
        # endregion

    # endregion

    # region call subfunction and add concat pixel values
    interp_bound_list = []
    for bound_line, line_type in zip(limit_bound_list, type_list):
        if line_type == "road-boundary":
            value = 255
        elif line_type == "lane-marking":
            value = 127
        interp_line = interpolate_boundary(bound_line)
        if interp_line is not None:
            value_vec = np.ones((1, interp_line.shape[1])) * value
            interp_bound_list.append(np.concatenate([interp_line, value_vec], axis=0))
        else:
            continue
    # endregion
    # endregion
    # print(f"Time for creating interpolation points:{time.time() - timer}")
    # timer = time.time()

    # region create image indexes
    interp_bound_arr = np.concatenate(interp_bound_list, axis=1)
    pixel_indexes = np.concatenate(
        [
            interp_bound_arr[0:2] // pixel_dist + res / 2,
            interp_bound_arr[2].reshape(1, interp_bound_arr.shape[1]),
        ],
        axis=0,
    )

    # endregion

    # region limit index indices to resolution
    pixel_indexes = np.delete(
        pixel_indexes,
        np.logical_or(
            np.amax(pixel_indexes[0:2], axis=0) > res - 1,
            np.amin(pixel_indexes[0:2], axis=0) < 0,
        ),
        axis=1,
    )

    # endregion

    # print(f"Time for creating index-set:{time.time() - timer}")
    # timer = time.time()

    # region build full-size image
    # create empty black image
    img = 0 * np.ones((res, res))
    pixel_values = pixel_indexes[2] if light_lane_dividers else 0
    # add values to image
    img[pixel_indexes[1].astype(int), pixel_indexes[0].astype(int)] = pixel_values
    # endregion
    # print(f"Time for building image:{time.time() - timer}")

    # saving the full size image needs less space than the pixel_index_data
    # there must be any kind of optimisation for saving pickling large tensors
    # in the background
    # pylint: disable=not-callable
    return img
    # pylint: enable=not-callable
    # endregion


def generate_scimg(
    lanelet_network, now_point, theta, time_step, watch_radius=64, draw_shape=True
):
    """Generate image input for neural network

    Arguments:
        scenario {[Commonroad scenario]} -- [Scenario object from CommonRoad]
        now_point {[list]} -- [[x,y] coordinates of vehicle right now that will be predicted]
        theta {[float]} -- [orientation of the vehicle that will be predicted]
        time_step {[float]} -- [Global time step of scenario]

    Keyword Arguments:
        draw_shape {bool} -- [Draw shapes of dynamic obstacles in image] (default: {True})

    Returns:
        img_gray [np.array] -- [Black and white image with 256 x 256 pixels of the scene]
    """
    my_dpi = 300
    draw_fig = plt.figure(
        figsize=(256 / my_dpi, 256 / my_dpi), dpi=my_dpi
    )  # 40 ms --> TODO shift outside the loop

    if theta > 2 * np.pi:
        theta -= 2 * np.pi
    elif theta < -(2 * np.pi):
        theta += 2 * np.pi

    lanelet_network.translate_rotate(np.array(-now_point), -theta)

    draw_params = {
        "time_begin": time_step,
        "lanelet_network": {"traffic_light": {"draw_traffic_lights": False}},
    }

    draw_object(
        lanelet_network,
        draw_params=draw_params,
    )

    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.gca().set_aspect("equal")
    plt.xlim(-watch_radius, watch_radius)
    plt.ylim(-watch_radius, watch_radius)

    draw_fig.canvas.draw()
    plt.close(draw_fig)

    # convert canvas to image
    img = np.fromstring(draw_fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
    img = img.reshape(draw_fig.canvas.get_width_height()[::-1] + (3,))

    img_gray = ~cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return img_gray


def generate_nbr_array(trans_traj_list, time_step, pp=30, window_size=[18, 78]):
    """Generates the array of trajectories around the vehicle being predicted

    Arguments:
        trans_traj_list {[type]} -- [description]
        time_step {[type]} -- [description]

    Keyword Arguments:
        pp {int} -- [description] (default: {31})
        window_size {list} -- [description] (default: {[18, 78]})

    Returns:
        [type] -- [description]
    """

    # Define window to identify neihbors
    r1 = [int(-i / 2) for i in window_size]  # [-9, -39]
    r2 = [int(i / 2) for i in window_size]

    nbrs = np.zeros((3, 13, pp, 2))
    pir_list = []
    for nbr in trans_traj_list:
        try:
            now_point_nbr = nbr[time_step]
        except IndexError:
            continue

        pir = point_in_rectangle(r1, r2, now_point_nbr)
        if pir:
            nbr_tmp = []
            for i in reversed(range(pp)):
                if time_step - i >= 0:
                    nbr_tmp.append(nbr[time_step - i])
                else:
                    nbr_tmp.append([np.nan, np.nan])

            nbrs[pir] = nbr_tmp
            pir_list.append(pir)

    return nbrs, pir_list, r1, r2
