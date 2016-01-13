__author__ = 'Steven Ogdahl'

from decaptcha import processing

steps = [
    ['invert'],
    ['separate_regions', {
        'regions': ((76, 0, 109, 50), (109, 0, 142, 50), (142, 0, 175, 50)),
        'separation': 5,
        'background': processing.WHITE
    }],
    ['remove_bbox', {
        'border': (70, 4, 70, 3),
        'background': processing.WHITE
    }],
    ['quantize', {
        'colors': 2,
        'palette': processing.PALETTE_IMAGE
    }],
    ['remove_lonely', {
        'max_neighbors': 4,
        'color': processing.BLACK
    }],
    ['remove_lonely', {
        'max_neighbors': 4,
        'color': processing.BLACK
    }],
    ['remove_lonely', {
        'max_neighbors': 4,
        'color': processing.BLACK
    }],
    ['remove_lonely', {
        'max_neighbors': 4,
        'color': processing.BLACK
    }],
    ['remove_juts', {
        'color': processing.BLACK
    }]
]

charset = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
