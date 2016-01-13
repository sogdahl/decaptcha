__author__ = 'Steven Ogdahl'

from decaptcha import processing

steps = [
    ['force_foreground'],
    ['remove_alpha', {
        'background': processing.WHITE
    }],
    ['remove_grid', {
        'keep_intersections': True
    }],
    ['separate_regions', {
        'regions': ((76, 0, 108, 50), (108, 0, 142, 50), (142, 0, 175, 50)),
        'separation': 5,
        'background': processing.WHITE
    }],
    ['quantize', {
        'colors': 2,
        'palette': processing.PALETTE_IMAGE
    }],
    ['stitch_orthogonal_gaps', {
        'color': processing.BLACK
    }],
    ['remove_lonely', {
        'max_neighbors': 1
    }],
    ['remove_juts', {
        'color': processing.BLACK
    }]
]

charset = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
