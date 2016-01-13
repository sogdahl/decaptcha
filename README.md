# decaptcha
Library for OCR processing CAPTCHA images


# Example
```
>>> from PIL import Image
>>> import decaptcha
>>> image = Image.open('images/sample_2__J53.jpg')
>>> decaptcha.perform_flow('sample_2', image)
'J53'
```