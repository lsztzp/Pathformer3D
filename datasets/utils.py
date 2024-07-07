import numpy as np


def mod(a, b):
    c = a // b
    r = a - c * b
    return r


def rotate_scanpath(lat_lon, angle):
    # 一条扫视路径，[N， 2]
    # angle: -180° ~ 120°

    # We convert [-180, 180] to [0, 360], then compute the new longitude.
    # We ``minus`` the angle here, which is different from what we do in rotating images,
    # because ffmepeg has a different coordination. For example, set ``yaw=60`` in ffmepg
    # equate to longitude = -60 + longitude
    new_lon = mod(lat_lon[:, 1] + 180 - angle, 360) - 180
    rotate_lat_lon = lat_lon
    rotate_lat_lon[:, 1] = new_lon
    return rotate_lat_lon


def handle_empty(sphere_coords, length):
    empty_index = np.where(sphere_coords[:, 0] == -999)[0]
    throw = False
    for _index in range(empty_index.shape[0]):
        # if not throw the scanpath of this user
        if not throw:
            # if the first one second is empty
            # 第一个点是无效点--取下一点（第二个点）值复制，若第二点也为空，则抛弃此条路径
            if empty_index[_index] == 0:
                # if the next second is not empty
                if sphere_coords[empty_index[_index] + 1, 0] != -999:
                    sphere_coords[empty_index[_index], 0] = sphere_coords[empty_index[_index] + 1, 0]
                    sphere_coords[empty_index[_index], 1] = sphere_coords[empty_index[_index] + 1, 1]
                else:
                    throw = True
                    # print(" Too many invalid gaze points !! {}".format(empty_index))

            # if the last one second is empty
            # 最后一个点为无效点则复制前一个点
            elif empty_index[_index] == (length - 1):
                sphere_coords[empty_index[_index], 0] = sphere_coords[empty_index[_index] - 1, 0]
                sphere_coords[empty_index[_index], 1] = sphere_coords[empty_index[_index] - 1, 1]

            # 前后两个点均为有效点，否则抛弃--使用两个点线性插值为当前点
            else:
                prev_x = sphere_coords[empty_index[_index] - 1, 1]
                prev_y = sphere_coords[empty_index[_index] - 1, 0]
                next_x = sphere_coords[empty_index[_index] + 1, 1]
                next_y = sphere_coords[empty_index[_index] + 1, 0]

                if prev_x == -999 or next_x == -999:
                    throw = True
                    # print(" Too many invalid gaze points !! {}".format(empty_index))

                else:
                    " Interpolate on lat "
                    sphere_coords[empty_index[_index], 0] = 0.5 * (prev_y + next_y)

                    " Interpolate on lon "
                    # the maximum distance between two points on a sphere is pi
                    if np.abs(next_x - prev_x) <= 180:
                        sphere_coords[empty_index[_index], 1] = 0.5 * (prev_x + next_x)
                    # jump to another side
                    else:
                        true_distance = 360 - np.abs(next_x - prev_x)
                        if next_x > prev_x:
                            _temp = prev_x - true_distance / 2
                            if _temp < -180:
                                _temp = 360 + _temp
                        else:
                            _temp = prev_x + true_distance / 2
                            if _temp > 180:
                                _temp = _temp - 360
                        sphere_coords[empty_index[_index], 1] = _temp

    return sphere_coords, throw


def sample_gaze_points(raw_data, length):
    # 一条扫视路径
    fixation_coords = []
    samples_per_bin = raw_data.shape[0] // length
    bins = raw_data[:samples_per_bin * length].reshape([length, -1, 2])
    for bin in range(length):
        " filter out invalid gaze points "
        _fixation_coords = bins[bin, np.where((bins[bin, :, 0] != 0) & (bins[bin, :, 1] != 0))]
        if _fixation_coords.shape[1] == 0:
            " mark the empty set"
            # [-999, -999] 标记无效点
            fixation_coords.append([-999, -999])
        else:
            " sample the first element in a set of one-second gaze points "
            sample_vale = _fixation_coords[0, 0]
            fixation_coords.append(sample_vale)
    sphere_coords = np.vstack(fixation_coords) - [90, 180]

    return sphere_coords
