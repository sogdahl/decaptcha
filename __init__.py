__author__ = 'Steven Ogdahl'

import importlib
import re

import processing
from utility.decaptcha import pytesser, processing


def perform_flow(flow, im, perform_ocr=True):
    flow_module = importlib.import_module('.flows.{0}'.format(flow), __name__)
    new_im = im
    for step in flow_module.steps:
        kwargs = {}
        if len(step) > 1:
            kwargs = step[1]
        new_im = getattr(processing, step[0])(new_im, **kwargs)

    if not new_im:
        return new_im
    elif perform_ocr:
        if hasattr(flow_module, 'charset'):
            charset = flow_module.charset
        else:
            charset = None

        text = pytesser.image_to_string(new_im, cleanup=True, charset=charset)
        return re.sub(r'\s', '', text)
    else:
        return new_im
