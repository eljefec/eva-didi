# Author: Ronny Restrepo
# URL: http://ronny.rest/blog/post_2017_03_26_lidar_birds_eye/
# URL: http://ronny.rest/blog/post_2017_03_27_lidar_height_slices/

from PIL import Image
import numpy as np

# ==============================================================================
#                                                                   SCALE_TO_255
# ==============================================================================
def scale_to_255(a, min, max, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - min) / float(max - min)) * 255).astype(dtype)


# ==============================================================================
#                                                          BIRDS_EYE_POINT_CLOUD
# ==============================================================================
def birds_eye_point_cloud(points,
                          side_range=(-10, 10),
                          fwd_range=(-10,10),
                          res=0.1,
                          min_height = -2.73,
                          max_height = 1.27,
                          saveto=None):
    """ Creates an 2D birds eye view representation of the point cloud data.
        You can optionally save the image to specified filename.

    Args:
        points:     (numpy array)
                    N rows of points data
                    Each point should be specified by at least 3 elements x,y,z
        side_range: (tuple of two floats)
                    (-left, right) in metres
                    left and right limits of rectangle to look at.
        fwd_range:  (tuple of two floats)
                    (-behind, front) in metres
                    back and front limits of rectangle to look at.
        res:        (float) desired resolution in metres to use
                    Each output pixel will represent an square region res x res
                    in size.
        min_height:  (float)(default=-2.73)
                    Used to truncate height values to this minumum height
                    relative to the sensor (in metres).
                    The default is set to -2.73, which is 1 metre below a flat
                    road surface given the configuration in the kitti dataset.
        max_height: (float)(default=1.27)
                    Used to truncate height values to this maximum height
                    relative to the sensor (in metres).
                    The default is set to 1.27, which is 3m above a flat road
                    surface given the configuration in the kitti dataset.
        saveto:     (str or None)(default=None)
                    Filename to save the image as.
                    If None, then it just displays the image.
    """
    x_lidar = points[:, 0]
    y_lidar = points[:, 1]
    z_lidar = points[:, 2]
    # r_lidar = points[:, 3]  # Reflectance

    # INDICES FILTER - of values within the desired rectangle
    # Note left side is positive y axis in LIDAR coordinates
    ff = np.logical_and((x_lidar > fwd_range[0]), (x_lidar < fwd_range[1]))
    ss = np.logical_and((y_lidar > -side_range[1]), (y_lidar < -side_range[0]))
    indices = np.argwhere(np.logical_and(ff,ss)).flatten()

    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (-y_lidar[indices]/res).astype(np.int32) # x axis is -y in LIDAR
    y_img = (x_lidar[indices]/res).astype(np.int32)  # y axis is -x in LIDAR
                                                     # will be inverted later

    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor used to prevent issues with -ve vals rounding upwards
    x_img -= int(np.floor(side_range[0]/res))
    y_img -= int(np.floor(fwd_range[0]/res))

    # CLIP HEIGHT VALUES - to between min and max heights
    pixel_values = np.clip(a = z_lidar[indices],
                           a_min=min_height,
                           a_max=max_height)

    # RESCALE THE HEIGHT VALUES - to be between the range 0-255
    pixel_values  = scale_to_255(pixel_values, min=min_height, max=max_height)

    # FILL PIXEL VALUES IN IMAGE ARRAY
    x_max = int((side_range[1] - side_range[0])/res)
    y_max = int((fwd_range[1] - fwd_range[0])/res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)
    im[-y_img, x_img] = pixel_values # -y because images start from top left

    # Convert from numpy array to a PIL image
    im = Image.fromarray(im)

    return im

# ==============================================================================
#                                                        BIRDS_EYE_HEIGHT_SLICES
# ==============================================================================
def birds_eye_height_slices(points,
                            n_slices=8,
                            height_range=(-2.73, 1.27),
                            side_range=(-10, 10),
                            fwd_range=(-10, 10),
                            res=0.1,
                            ):
    """ Creates an array that is a birds eye view representation of the
        reflectance values in the point cloud data, separated into different
        height slices.

    Args:
        points:     (numpy array)
                    Nx4 array of the points cloud data.
                    N rows of points. Each point represented as 4 values,
                    x,y,z, reflectance
        n_slices :  (int)
                    Number of height slices to use.
        height_range: (tuple of two floats)
                    (min, max) heights (in metres) relative to the sensor.
                    The slices calculated will be within this range, plus
                    two additional slices for clipping all values below the
                    min, and all values above the max.
                    Default is set to (-2.73, 1.27), which corresponds to a
                    range of -1m to 3m above a flat road surface given the
                    configuration of the sensor in the Kitti dataset.
        side_range: (tuple of two floats)
                    (-left, right) in metres
                    Left and right limits of rectangle to look at.
                    Defaults to 10m on either side of the car.
        fwd_range:  (tuple of two floats)
                    (-behind, front) in metres
                    back and front limits of rectangle to look at.
                    Defaults to 10m behind and 10m in front.
        res:        (float) desired resolution in metres to use
                    Each output pixel will represent an square region res x res
                    in size along the front and side plane.
    """
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]
    r_points = points[:, 3]  # Reflectance

    # FILTER INDICES - of only the points within the desired rectangle
    # Note left side is positive y axis in LIDAR coordinates
    ff = np.logical_and((x_points > fwd_range[0]), (x_points < fwd_range[1]))
    ss = np.logical_and((y_points > -side_range[1]), (y_points < -side_range[0]))
    indices = np.argwhere(np.logical_and(ff, ss)).flatten()

    # KEEPERS - The actual points that are within the desired  rectangle
    y_points = y_points[indices]
    x_points = x_points[indices]
    z_points = z_points[indices]
    r_points = r_points[indices]

    # CONVERT TO PIXEL POSITION VALUES - Based on resolution
    x_img = (-y_points / res).astype(np.int32) # x axis is -y in LIDAR
    y_img = (x_points / res).astype(np.int32)  # y axis is -x in LIDAR
                                               # direction to be inverted later
    # SHIFT PIXELS TO HAVE MINIMUM BE (0,0)
    # floor used to prevent issues with -ve vals rounding upwards
    x_img -= int(np.floor(side_range[0] / res))
    y_img -= int(np.floor(fwd_range[0] / res))

    # ASSIGN EACH POINT TO A HEIGHT SLICE
    # n_slices-1 is used because values above max_height get assigned to an
    # extra index when we call np.digitize().
    bins = np.linspace(height_range[0], height_range[1], num=n_slices-1)
    slice_indices = np.digitize(z_points, bins=bins, right=False)

    # RESCALE THE REFLECTANCE VALUES - to be between the range 0-255
    pixel_values = scale_to_255(r_points, min=0.0, max=1.0)

    # FILL PIXEL VALUES IN IMAGE ARRAY
    # -y is used because images start from top left
    x_max = int((side_range[1] - side_range[0]) / res)
    y_max = int((fwd_range[1] - fwd_range[0]) / res)
    im = np.zeros([y_max, x_max, n_slices], dtype=np.uint8)
    im[-y_img, x_img, slice_indices] = pixel_values

    return im
