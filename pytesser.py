__author__ = 'Steven Ogdahl'
"""OCR in Python using the Tesseract engine from Google
http://code.google.com/p/pytesser/
by Michael J.T. O'Kelly
V 0.0.1, 3/10/07"""

import os
import subprocess

from PIL import Image

from utility.decaptcha import util
import errors
import settings


scratch_image_name = "temp.png" # This file must be .bmp or other Tesseract-compatible format
scratch_text_name_root = "temp" # Leave out the .txt extension
scratch_charset = 'charset'
cleanup_scratch_flag = True  # Temporary files cleaned up after OCR operation

def call_tesseract(input_filename, output_filename, charset=None):
    """Calls external tesseract on input file (restrictions on types),
    outputting output_filename + '.txt'
    Optionally, you can provide a character set to limit the OCR characters
    the ones specified"""
    devnull = open(os.devnull, 'w')
    args = [settings.TESSERACT_PATH, input_filename, output_filename]

    if charset:
        util.charset_to_scratch(charset, scratch_charset)
        args.append(scratch_charset)

    proc = subprocess.Popen(args, stderr=devnull)
    retcode = proc.wait()
    if retcode != 0:
        errors.check_for_errors()

def image_to_string(im, cleanup=cleanup_scratch_flag, charset=None):
    """Converts im to file, applies tesseract, and fetches resulting text.
    If cleanup=True, delete scratch files after operation."""
    try:
        util.image_to_scratch(im, scratch_image_name)
        call_tesseract(scratch_image_name, scratch_text_name_root, charset=charset)
        text = util.retrieve_text(scratch_text_name_root)
    finally:
        if cleanup:
            util.perform_cleanup(scratch_image_name, scratch_text_name_root, scratch_charset)
    return text

def image_file_to_string(filename, cleanup=cleanup_scratch_flag, graceful_errors=True, charset=None):
    """Applies tesseract to filename; or, if image is incompatible and graceful_errors=True,
    converts to compatible format and then applies tesseract.  Fetches resulting text.
    If cleanup=True, delete scratch files after operation."""
    try:
        try:
            call_tesseract(filename, scratch_text_name_root, charset=charset)
            text = util.retrieve_text(scratch_text_name_root)
        except errors.Tesser_General_Exception:
            if graceful_errors:
                im = Image.open(filename)
                text = image_to_string(im, cleanup)
            else:
                raise
    finally:
        if cleanup:
            util.perform_cleanup(scratch_image_name, scratch_text_name_root, scratch_charset)
    return text


if __name__ == '__main__':
    im = Image.open('phototest.tif')
    text = image_to_string(im)
    print text
    try:
        text = image_file_to_string('fnord.tif', graceful_errors=False)
    except errors.Tesser_General_Exception, value:
        print "fnord.tif is incompatible filetype.  Try graceful_errors=True"
        print value
    text = image_file_to_string('fnord.tif', graceful_errors=True)
    print "fnord.tif contents:", text
    text = image_file_to_string('fonts_test.png', graceful_errors=True)
    print text

