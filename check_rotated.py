import math
import numpy as np

def rotate(origin, point, radian):
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(radian) * (px - ox) - math.sin(radian) * (py - oy)
    qy = oy + math.sin(radian) * (px - ox) + math.cos(radian) * (py - oy)
    return round(qx), round(qy)


def rotate_box_dot(x_cen, y_cen, width, height, theta):

    x_min = x_cen - width / 2
    y_min = y_cen - height / 2
    print(f"x_min : {x_min} | y_min : {y_min} | width : {width} | height : {height}")
    rotated_x1, rotated_y1 = rotate((x_cen, y_cen), (x_min, y_min), theta)
    rotated_x2, rotated_y2 = rotate((x_cen, y_cen), (x_min, y_min + height), theta)
    rotated_x3, rotated_y3 = rotate(
        (x_cen, y_cen), (x_min + width, y_min + height), theta
    )
    rotated_x4, rotated_y4 = rotate((x_cen, y_cen), (x_min + width, y_min), theta)

    answer_dict_ = {
        "Rx": np.array([rotated_x1, rotated_x2, rotated_x3, rotated_x4]),
        "Ry": np.array([rotated_y1, rotated_y2, rotated_y3, rotated_y4]),
    }

    return answer_dict_

if __name__ == "__main__":
    # cx, cy, width, height, theta = rbbox
    width = 1861-1790
    height = 407-223
    cx = 1790 + width/2
    cy = 223 + height/2
    # width = 1854-1777
    # height = 398-196
    # cx = 1777+width/2
    # cy = 196+height/2
    radian = 0.3514
    result = rotate_box_dot(cx, cy, width, height, radian)
    print(result)