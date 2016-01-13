__author__ = 'Steven Ogdahl'

import colorsys
import math
from PIL import Image, ImageDraw, ImageChops, ImageOps, ImageStat, ImageFilter
import random

PALETTE = [
      0,   0,   0,
    255, 255, 255
] + [0, ] * 254 * 3
PALETTE_IMAGE = Image.new("P", (1, 1), 0)
PALETTE_IMAGE.putpalette(PALETTE)

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

def _rgb_to_hsv(r, g, b):
    return colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)

def _invert_color(color):
    if isinstance(color, tuple):
        hsv = _rgb_to_hsv(*color)
        return tuple([int(255 * c) for c in colorsys.hsv_to_rgb(hsv[0], hsv[1], 1 - hsv[2])])
    else:
        return 1 - color


def _color_diff(p1, p2):
    if isinstance(p1, tuple):
        hsv1 = _rgb_to_hsv(*p1[:3])
    else:
        hsv1 = (0, 0, p1)
    if isinstance(p2, tuple):
        hsv2 = _rgb_to_hsv(*p2[:3])
    else:
        hsv2 = (0, 0, p2)
    return abs(hsv1[2] - hsv2[2])


def _get_nxn_grid(im, x, y, n=3, background=WHITE):
    grid = [range(n) for _ in xrange(n)]
    for y_i in xrange(n):
        for x_i in xrange(n):
            p = x + x_i - 2, y + y_i - 2
            if p[0] < 0 or p[1] < 0 or p[0] >= im.size[0] or p[1] >= im.size[1]:
                grid[p[0]][p[1]] = background
            else:
                grid[p[0]][p[1]] = im.getpixel(p)
    return grid


def _is_color_proximal(color, method, compare, proximity):
    if method == 'hue':
        return abs(compare - _rgb_to_hsv(*color)[0]) <= proximity
    return False


def quantize(im, colors=2, method=None, kmeans=0, palette=None):
    return im.quantize(colors=colors, method=method, kmeans=kmeans, palette=palette)


def force_foreground(im, threshold=0.5):
    imagestat = ImageStat.Stat(im)
    # Assuming greyscale image
    hsv_image = _rgb_to_hsv(*imagestat.rms[:3])
    if hsv_image[2] < threshold:
        return invert(im)
    return im


def remove_alpha(im, background=WHITE):
    if im.mode == "RGBA":
        im.load()
        new_im = Image.new("RGB", im.size, background)
        new_im.paste(im, mask=im.split()[3]) # Index 3 is the alpha channel
        return new_im
    return im


def remove_bbox(im, border, background=WHITE):
    im_copy = im.copy()
    im_copy = ImageOps.crop(im_copy, border=border)
    return ImageOps.expand(im_copy, border=border, fill=background)


def center_data(im, auto=True):
    im_copy = im.copy()
    image_center = (im_copy.size[0] / 2, im_copy.size[1] / 2)

    bbox = im_copy.getbbox()
    if auto and bbox == (0, 0) + im_copy.size:
        im_invert = invert(im_copy)
        bbox = im_invert.getbbox()

    bbox_center = ((bbox[2] - bbox[0]) / 2, (bbox[3] - bbox[1]) / 2)
    im_copy = ImageChops.offset(im_copy, xoffset=(image_center[0] - bbox_center[0]) / 2, yoffset=(image_center[1] - bbox_center[1]) / 2)
    return im_copy


def filter(im, filter):
    if filter == 'blur':
        filter = ImageFilter.BLUR
    elif filter == 'contour':
        filter = ImageFilter.CONTOUR
    elif filter == 'detail':
        filter = ImageFilter.DETAIL
    elif filter == 'edge_enhance':
        filter = ImageFilter.EDGE_ENHANCE
    elif filter == 'edge_enhance_more':
        filter = ImageFilter.EDGE_ENHANCE_MORE
    elif filter == 'emboss':
        filter = ImageFilter.EMBOSS
    elif filter == 'find_edges':
        filter = ImageFilter.FIND_EDGES
    elif filter == 'smooth':
        filter = ImageFilter.SMOOTH
    elif filter == 'smooth_more':
        filter = ImageFilter.SMOOTH_MORE
    elif filter == 'sharpen':
        filter = ImageFilter.SHARPEN

    return im.filter(filter)


def separate_regions(im, regions, separation, background=WHITE):
    pieces = []
    num_regions = len(regions)
    center_region = (num_regions + 1) / 2.0 - 1
    for i in range(num_regions):
        region = regions[i]
        piece = remove_bbox(im, (region[0], region[1], im.size[0] - region[2], im.size[1] - region[3]), background=background)
        piece = ImageChops.offset(piece, xoffset=int(round((i - center_region) * separation)), yoffset=0)
        pieces.append(piece)

    while len(pieces) > 1:
        pieces = [ImageChops.darker(pieces[0], pieces[1])] + pieces[2:]
    return pieces[0]


def remove_grid(im, keep_intersections=False, threshold=0.5):
    # Let's start with grid detection
    p1 = im.getpixel((0, 0))
    p2 = im.getpixel((1, 1))

    horizontal_line_columns = []
    vertical_line_rows = []

    background_color, foreground_color = p1, p2
    offset = (0, 0)
    # This grid starts at 0, 0
    if _color_diff(p1, p2) > threshold:
        background_color, foreground_color = p2, p1
        offset = (1, 1)
        horizontal_line_columns.append(0)
        vertical_line_rows.append(0)

    col = offset[0]
    while col < im.size[0]:
        if _color_diff(im.getpixel((col, offset[1])), background_color) > threshold and \
                        _color_diff(im.getpixel((col, im.size[1] - offset[1])), background_color) > threshold:
            horizontal_line_columns.append(col)
        col += 1

    row = offset[1]
    while row < im.size[1]:
        if _color_diff(im.getpixel((offset[0], row)), background_color) > threshold and \
                        _color_diff(im.getpixel((im.size[0] - offset[0], row)), background_color) > threshold:
            vertical_line_rows.append(row)
        row += 1

    im_copy = im.copy()
    draw = ImageDraw.Draw(im_copy)

    # Remove columns
    for col in horizontal_line_columns:
        draw.line((col, 0, col, im_copy.size[1] - 1), fill=background_color)

    # Remove rows
    for row in vertical_line_rows:
        draw.line((0, row, im_copy.size[0] - 1, row), fill=background_color)

    if keep_intersections:
        # Paradoxically, add back in the intersections between rows & columns
        for col in horizontal_line_columns:
            for row in vertical_line_rows:
                draw.point((col, row), fill=foreground_color)

    return im_copy


def stitch_orthogonal_gaps(im, color=BLACK, gapsize=1, threshold=0.5):
    im_copy = im.copy()
    for y in xrange(im_copy.size[1]):
        for x in xrange(im_copy.size[0]):
            pixel = im.getpixel((x, y))
            if _color_diff(pixel, color) > threshold:
                continue
            if x >= 2: # Check left
                if _color_diff(pixel, im.getpixel((x - 1, y))) > threshold and \
                                _color_diff(pixel, im.getpixel((x - 2, y))) < threshold:
                    im_copy.putpixel((x - 1, y), pixel)
            if y >= 2: # Check up
                if _color_diff(pixel, im.getpixel((x, y - 1))) > threshold and \
                                _color_diff(pixel, im.getpixel((x, y - 2))) < threshold:
                    im_copy.putpixel((x, y - 1), pixel)
            if x <= im.size[0] - 3: # Check right
                if _color_diff(pixel, im.getpixel((x + 1, y))) > threshold and \
                                _color_diff(pixel, im.getpixel((x + 2, y))) < threshold:
                    im_copy.putpixel((x + 1, y), pixel)
            if y <= im.size[1] - 3: # Check down
                if _color_diff(pixel, im.getpixel((x, y + 1))) > threshold and \
                                _color_diff(pixel, im.getpixel((x, y + 2))) < threshold:
                    im_copy.putpixel((x, y + 1), pixel)
    return im_copy

def stitch_kissing_corners(im, color=BLACK, threshold=0.5):
    im_copy = im.copy()
    for y in xrange(1, im_copy.size[1] - 1):
        for x in xrange(1, im_copy.size[0] - 1):
            pixel = im.getpixel((x, y))
            if _color_diff(pixel, color) > threshold:
                continue

            colordiffs = [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ]

            for y_i in xrange(-1, 2):
                for x_i in xrange(-1, 2):
                    if (x_i, y_i) == (0, 0):
                        continue
                    colordiffs[y_i + 1][x_i + 1] = _color_diff(pixel, im.getpixel((x + x_i, y + y_i)))

            # Check topleft / bottomright
            if colordiffs[0][0] <= threshold and colordiffs[0][1] > threshold and colordiffs[0][2] > threshold and \
                colordiffs[1][0] > threshold and '                          ' and colordiffs[1][2] > threshold and \
                colordiffs[2][0] > threshold and colordiffs[2][1] > threshold and colordiffs[2][2] <= threshold:
                # If so, color the 4 orthogonally-adjacent pixels
                im_copy.putpixel((x - 1, y), pixel)
                im_copy.putpixel((x + 1, y), pixel)
                im_copy.putpixel((x, y - 1), pixel)
                im_copy.putpixel((x, y + 1), pixel)

            # Check topright / bottomleft
            if colordiffs[0][0] > threshold and colordiffs[0][1] > threshold and colordiffs[0][2] <= threshold and \
                colordiffs[1][0] > threshold and '                          ' and colordiffs[1][2] > threshold and \
                colordiffs[2][0] <= threshold and colordiffs[2][1] > threshold and colordiffs[2][2] > threshold:
                # If so, color the 4 orthogonally-adjacent pixels
                im_copy.putpixel((x - 1, y), pixel)
                im_copy.putpixel((x + 1, y), pixel)
                im_copy.putpixel((x, y - 1), pixel)
                im_copy.putpixel((x, y + 1), pixel)
    return im_copy

def remove_lonely(im, max_neighbors=0, threshold=0.5, color=None):
    im_copy = im.copy()
    draw = ImageDraw.Draw(im_copy)
    for y in xrange(im_copy.size[1]):
        for x in xrange(im_copy.size[0]):
            neighbors = []
            pixel = im.getpixel((x, y))
            if color and _color_diff(pixel, color) > threshold:
                continue

            for y_i in xrange(-1, 2):
                for x_i in xrange(-1, 2):
                    neighbor = (x + x_i, y + y_i)
                    if (x + x_i, y + y_i) == (x, y):
                        continue
                    if neighbor[0] >= 0 and neighbor[0] < im.size[0] and neighbor[1] >= 0 and neighbor[1] < im.size[1]:
                        if _color_diff(pixel, im.getpixel(neighbor)) < threshold:
                            neighbors.append(neighbor)
            if len(neighbors) <= max_neighbors:
                draw.point((x, y), fill=_invert_color(pixel))

    return im_copy

def remove_juts(im, color=BLACK, threshold=0.5):
    im_copy = im.copy()
    for y in xrange(im_copy.size[1]):
        for x in xrange(im_copy.size[0]):
            pixel = im.getpixel((x, y))
            if _color_diff(pixel, color) > threshold:
                continue

            colordiffs = [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]
            ]

            for y_i in xrange(-1, 2):
                for x_i in xrange(-1, 2):
                    if (x_i, y_i) == (0, 0):
                        continue
                    if x + x_i < 0 or x + x_i >= im.size[0] or \
                        y + y_i < 0 or y + y_i >= im.size[1]:
                        continue

                    colordiffs[y_i + 1][x_i + 1] = _color_diff(pixel, im.getpixel((x + x_i, y + y_i)))

            # Check left edge
            if colordiffs[0][0] <= threshold and colordiffs[0][1] > threshold and colordiffs[0][2] > threshold and \
                colordiffs[1][0] <= threshold and '                          ' and colordiffs[1][2] > threshold and \
                colordiffs[2][0] <= threshold and colordiffs[2][1] > threshold and colordiffs[2][2] > threshold:
                im_copy.putpixel((x, y), _invert_color(pixel))

            # Check top edge
            if colordiffs[0][0] <= threshold and colordiffs[0][1] <= threshold and colordiffs[0][2] <= threshold and \
                colordiffs[1][0] > threshold and '                          ' and colordiffs[1][2] > threshold and \
                colordiffs[2][0] > threshold and colordiffs[2][1] > threshold and colordiffs[2][2] > threshold:
                im_copy.putpixel((x, y), _invert_color(pixel))

            # Check right edge
            if colordiffs[0][0] > threshold and colordiffs[0][1] > threshold and colordiffs[0][2] <= threshold and \
                colordiffs[1][0] > threshold and '                          ' and colordiffs[1][2] <= threshold and \
                colordiffs[2][0] > threshold and colordiffs[2][1] > threshold and colordiffs[2][2] <= threshold:
                im_copy.putpixel((x, y), _invert_color(pixel))

            # Check bottom edge
            if colordiffs[0][0] > threshold and colordiffs[0][1] > threshold and colordiffs[0][2] > threshold and \
                colordiffs[1][0] > threshold and '                          ' and colordiffs[1][2] > threshold and \
                colordiffs[2][0] <= threshold and colordiffs[2][1] <= threshold and colordiffs[2][2] <= threshold:
                im_copy.putpixel((x, y), _invert_color(pixel))

    return im_copy

def invert(im):
    return ImageOps.invert(im)

def threshold(im, threshold=0.5):
    im_copy = im.copy()
    for y in xrange(im_copy.size[1]):
        for x in xrange(im_copy.size[0]):
            if _rgb_to_hsv(*im_copy.getpixel((x, y))[:3])[2] >= threshold:
                im_copy.putpixel((x, y), WHITE)
            else:
                im_copy.putpixel((x, y), BLACK)
    return im_copy

def fill_shapes(im, fill_holes=False, separation=5, edge_depth=2, minimum_size=15):
    im_copy = im.copy()
    used_hues = set()
    color_histogram = {}
    color_extent = {}
    edge_colors = set()
    points_checked = set()

    draw = ImageDraw.Draw(im_copy)

    def _walk_path(origin, color, process_diagonals, edge_depth):
        checked_points = set()
        check_queue = []
        check_queue.insert(0, origin)
        while check_queue:
            point = check_queue.pop()
            if point not in checked_points:
                checked_points.add(point)
                if _rgb_to_hsv(*im_copy.getpixel(point))[2] <= 0.5:
                    draw.point(point, fill=color)
                    color_histogram[color] = color_histogram.get(color, 0) + 1
                    if color not in color_extent:
                        color_extent[color] = [origin[0], origin[1], origin[0], origin[1]]
                    if point[0] < color_extent[color][0]:
                        color_extent[color][0] = point[0]
                    elif point[0] > color_extent[color][2]:
                        color_extent[color][2] = point[0]
                    if point[1] < color_extent[color][1]:
                        color_extent[color][1] = point[1]
                    elif point[1] > color_extent[color][3]:
                        color_extent[color][3] = point[1]
                    if (point[0] < edge_depth or point[0] >= im_copy.size[0] - edge_depth or
                                point[1] < edge_depth or point[1] >= im_copy.size[1] - edge_depth) \
                            and color not in edge_colors:
                        edge_colors.add(color)
                    if point[0] > 0:
                        check_queue.insert(0, (point[0] - 1, point[1]))
                    if point[1] > 0:
                        check_queue.insert(0, (point[0], point[1] - 1))
                    if point[0] < im_copy.size[0] - 1:
                        check_queue.insert(0, (point[0] + 1, point[1]))
                    if point[1] < im_copy.size[1] - 1:
                        check_queue.insert(0, (point[0], point[1] + 1))
                    if process_diagonals:
                        if point[0] > 0:
                            if point[1] > 0:
                                check_queue.insert(0, (point[0] - 1, point[1] - 1))
                            if point[1] < im_copy.size[1] - 1:
                                check_queue.insert(0, (point[0] - 1, point[1] + 1))
                        if point[0] < im_copy.size[0] - 1:
                            if point[1] > 0:
                                check_queue.insert(0, (point[0] + 1, point[1] - 1))
                            if point[1] < im_copy.size[1] - 1:
                                check_queue.insert(0, (point[0] + 1, point[1] + 1))
        return

    def _region_points(point, threshold, max_size):
        checked_points = set()
        check_queue = []
        check_queue.insert(0, point)
        points = []
        hue = _rgb_to_hsv(*im_copy.getpixel(point))[0]
        while check_queue:
            p = check_queue.pop()
            if p not in checked_points:
                checked_points.add(p)
                points.append(p)
                if len(points) > max_size:
                    return []
                if p[0] >= 1 and _is_color_proximal(im_copy.getpixel((p[0] - 1, p[1])), 'hue', hue, threshold):
                    check_queue.insert(0, (p[0] - 1, p[1]))
                if p[0] + 1 < im_copy.size[0] and _is_color_proximal(im_copy.getpixel((p[0] + 1, p[1])), 'hue', hue, threshold):
                    check_queue.insert(0, (p[0] + 1, p[1]))
                if p[1] >= 1 and _is_color_proximal(im_copy.getpixel((p[0], p[1] - 1)), 'hue', hue, threshold):
                    check_queue.insert(0, (p[0], p[1] - 1))
                if p[1] + 1 < im_copy.size[1] and _is_color_proximal(im_copy.getpixel((p[0], p[1] + 1)), 'hue', hue, threshold):
                    check_queue.insert(0, (p[0], p[1] + 1))
        return points

    for y in xrange(im_copy.size[1]):
        for x in xrange(im_copy.size[0]):
            pixel = im_copy.getpixel((x, y))
            pixel_hsv = _rgb_to_hsv(*pixel)
            if pixel_hsv[2] < 0.5:
                hue = -1.0
                counter = 0
                while (hue < 0) and counter < 500:
                    counter += 1
                    hue = random.random()
                    for used_hue in used_hues:
                        if abs(used_hue - hue) < 0.015:
                            hue = -1.0
                used_hues.add(hue)
                color = tuple([int(255 * c) for c in colorsys.hsv_to_rgb(hue, 0.75, 0.65)])
                _walk_path((x, y), color, True, edge_depth)

    background = WHITE
    to_remove = set()
    center = (im_copy.size[0] / 2, im_copy.size[1] / 2)
    root_point_five = math.sqrt(0.5)
    for (color, extent) in color_extent.items():
        distances = (
                math.sqrt(math.pow((extent[1] - center[1]) / float(im_copy.size[1]), 2.0) + math.pow((extent[0] - center[0]) / float(im_copy.size[0]), 2.0)) / root_point_five,
                math.sqrt(math.pow((extent[1] - center[1]) / float(im_copy.size[1]), 2.0) + math.pow((extent[2] - center[0]) / float(im_copy.size[0]), 2.0)) / root_point_five,
                math.sqrt(math.pow((extent[3] - center[1]) / float(im_copy.size[1]), 2.0) + math.pow((extent[0] - center[0]) / float(im_copy.size[0]), 2.0)) / root_point_five,
                math.sqrt(math.pow((extent[3] - center[1]) / float(im_copy.size[1]), 2.0) + math.pow((extent[2] - center[0]) / float(im_copy.size[0]), 2.0)) / root_point_five,
                abs((extent[2] + extent[0]) / 2.0 - center[0]) / float(im_copy.size[0]),
                abs((extent[3] + extent[1]) / 2.0 - center[1]) / float(im_copy.size[1]),
                math.sqrt(math.pow((extent[3] + extent[1]) / 2.0 - center[1], 2.0) + math.pow((extent[2] + extent[0]) / 2 - center[0], 2.0))
        )
        if (distances[0] > 0.7 and distances[1] > 0.7 and distances[2] > 0.7 and distances[3] > 0.7) \
                or distances[4] > 0.5 or distances[5] > 0.4:
            to_remove.add(color)

    for y in xrange(im_copy.size[1]):
        for x in xrange(im_copy.size[0]):
            pixel = im_copy.getpixel((x, y))
            if pixel in to_remove or \
                    (minimum_size > 0 and pixel in color_histogram and color_histogram[pixel] < minimum_size) or \
                    (pixel in edge_colors and pixel in color_histogram and color_histogram[pixel] < 300):
                draw.point((x, y), fill=background)
                to_remove.add(pixel)

    for color in to_remove:
        color_histogram.pop(color, None)
        color_extent.pop(color, None)
        edge_colors.discard(color)
    to_remove.clear()

    for (color_shape, shape_bbox) in color_extent.items():
        for (color_surround, surround_bbox) in color_extent.items():
            if color_shape == color_surround:
                continue
            if surround_bbox[0] <= shape_bbox[0] and surround_bbox[1] <= shape_bbox[1] and \
                            surround_bbox[2] >= shape_bbox[2] and surround_bbox[3] >= shape_bbox[3]:
                for y in xrange(shape_bbox[1], shape_bbox[3] + 1):
                    for x in xrange(shape_bbox[0], shape_bbox[2] + 1):
                        if im_copy.getpixel((x, y)) == color_shape:
                            draw.point((x, y), fill=color_surround)
                to_remove.add(color_shape)
                break

    for color in to_remove:
        color_histogram.pop(color, None)
        color_extent.pop(color, None)
        edge_colors.discard(color)
    to_remove.clear()

    if fill_holes:
        max_size = 0
        for y in xrange(im_copy.size[1]):
            for x in xrange(im_copy.size[0]):
                pixel = im_copy.getpixel((x, y))
                if pixel in color_histogram:
                    if x > 1:
                        pixel_i = im_copy.getpixel((x - 1, y))
                        if pixel_i not in color_histogram:
                            points = _region_points((x - 1, y), 0.05, 350)
                            if len(points) < 15:
                                for p in points:
                                    draw.point(p, fill=pixel)
                                max_size = max(max_size, len(points))
                    if y > 1:
                        pixel_j = im_copy.getpixel((x, y - 1))
                        if pixel_j not in color_histogram:
                            points = _region_points((x, y - 1), 0.05, 350)
                            if len(points) < 15:
                                for p in points:
                                    draw.point(p, fill=pixel)
                                max_size = max(max_size, len(points))

    if separation > 0:
        return separate_shapes(im_copy, separation, color_extent)

    return im_copy


def separate_shapes(im, separation, color_bboxes):
    im_copy = im.copy()
    center = (im_copy.size[0] / 2, im_copy.size[1] / 2)
    closest_center = -im.size[0]
    centers = {}
    for (color, bbox) in color_bboxes.items():
        distance = (bbox[2] + bbox[0]) / 2.0 - center[0]
        if abs(distance) < abs(closest_center):
            closest_center = distance
        if distance not in centers:
            centers[distance] = set()
        centers[distance].add(color)

    background = WHITE

    keys = centers.keys()
    keys.sort()
    center_index = keys.index(closest_center)
    offsets = []
    for i in xrange(len(keys)):
        offsets.append(0)
    shift = 0
    for index in xrange(center_index + 1, len(keys)):
        if keys[index] - keys[index - 1] >= 5:
            shift += 1
        offsets[index] = shift
    shift = 0
    for index in xrange(center_index - 1, -1, -1):
        if keys[index + 1] - keys[index] >= 5:
            shift -= 1
        offsets[index] = shift

    draw = ImageDraw.Draw(im_copy)

    for index in xrange(center_index):
        x_shift = offsets[index] * separation
        for color in centers[keys[index]]:
            for x in xrange(color_bboxes[color][0], color_bboxes[color][2] + 1):
                for y in xrange(color_bboxes[color][1], color_bboxes[color][3] + 1):
                    pixel = im_copy.getpixel((x, y))
                    if pixel == color:
                        draw.point((x + x_shift, y), fill=color)
                        draw.point((x, y), fill=background)
    for index in xrange(len(keys) - 1, center_index, -1):
        x_shift = offsets[index] * separation
        for color in centers[keys[index]]:
            for x in xrange(color_bboxes[color][2], color_bboxes[color][0] - 1, -1):
                for y in xrange(color_bboxes[color][1], color_bboxes[color][3] + 1):
                    pixel = im_copy.getpixel((x, y))
                    if pixel == color:
                        draw.point((x + x_shift, y), fill=color)
                        draw.point((x, y), fill=background)

    return im_copy


def decide_flow(im, flows):
    from . import perform_flow
    histogram = im.histogram()
    red_histogram = histogram[:256]
    black_count = red_histogram[0]
    white_count = red_histogram[255]
    other_count = sum(red_histogram[1:-1])
    total_count = float(black_count + white_count + other_count)

    for flow_name, flow_parameters in flows.items():
        matching = True
        if 'histogram' in flow_parameters:
            for hist_type, hist_range in flow_parameters['histogram'].items():
                if hist_type == 'black':
                    hist_value = black_count / total_count
                elif hist_type == 'white':
                    hist_value = white_count / total_count
                elif hist_type == 'other':
                    hist_value = other_count / total_count
                else:
                    continue
                if hist_value < hist_range[0] or hist_value > hist_range[1]:
                    matching = False
                    break
        if matching:
            return perform_flow(flow_name, im, perform_ocr=False)

    # No matching flows found, so return None (no processing done)
    return None

processors = [quantize, force_foreground, remove_alpha, remove_bbox, center_data, filter, separate_regions, remove_grid,
              stitch_orthogonal_gaps, stitch_kissing_corners, remove_lonely, remove_juts, invert, threshold,
              fill_shapes, separate_shapes, decide_flow]
