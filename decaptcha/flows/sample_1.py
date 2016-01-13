__author__ = 'Steven Ogdahl'

from decaptcha import processing

steps = [
    ['force_foreground'],
    ['remove_alpha', {
        'background': processing.WHITE
    }],
    ['threshold', {
        'threshold': 0.75
    }],
    ['remove_grid', {
        'keep_intersections': True
    }],
    ['stitch_orthogonal_gaps', {
        'color': processing.BLACK
    }],
    ['remove_lonely', {
        'max_neighbors': 1
    }],
    ['stitch_kissing_corners', {
        'color': processing.BLACK
    }],
    ['remove_juts', {
        'color': processing.BLACK
    }],
    ['stitch_orthogonal_gaps', {
        'color': processing.BLACK
    }],
    ['center_data'],
    ['fill_shapes', {
        'fill_holes': False,
        'separation': 4,
        'edge_depth': 2,
        'minimum_size': 15
    }],
    ['threshold', {
        'threshold': 0.75
    }]
]

charset = 'ABCDEF0123456789'
