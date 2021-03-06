"""
Augmenters that apply affine transformations.

To use the augmenters, clone the complete repo and use
`from vidaug import augmenters as va`
and then e.g. :
    seq = va.Sequential([ va.RandomRotate(30),
                          va.RandomResize(0.2)  ])

List of augmenters:
    * RandomRotate
    * RandomResize
    * RandomTranslate
    * RandomShear
"""

import numpy as np
import numbers
import random
import PIL
import cv2


class RandomRotate(object):
    """
    Rotate video randomly by a random angle within given boundsi.

    Args:
        degrees (sequence or int): Range of degrees to randomly
        select from. If degrees is a number instead of sequence
        like (min, max), the range of degrees, will be
        (-degrees, +degrees).
    """

    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError('If degrees is a single number,'
                                 'must be positive')
            degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError('If degrees is a sequence,'
                                 'it must be of len 2.')

        self.degrees = degrees

    def __call__(self, clip):
        angle = random.uniform(self.degrees[0], self.degrees[1])
        is_PIL = isinstance(clip[0], PIL.Image.Image)
        if is_PIL:
            clip = [np.asarray(img) for img in clip]

        im_h = clip[0].shape[0]
        im_w = clip[0].shape[1]
        (cX, cY) = (im_w // 2, im_h // 2)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # compute the new bounding dimensions of the image
        nW = int((im_h*sin)+(im_w*cos))
        nH = int((im_h*cos)+(im_w*sin))
        M[0, 2] += (nW / 2)-cX
        M[1, 2] += (nH / 2)-cY
        data_final = []
        for image in clip:
            new_image = cv2.warpAffine(image, M, (nW, nH))
            top_x = (nW-im_w) // 2
            top_y = (nH-im_h) // 2
            cropped_image = new_image[top_y:top_y+im_h, top_x:top_x+im_w]
            data_final.append(cropped_image)

        if is_PIL:
            return [PIL.Image.fromarray(img) for img in data_final]
        else:
            return data_final


class RandomResize(object):
    """
    Resize video by zooming in and out.

    Args:
        rate (float): Video is scaled uniformly between
        [1-rate, 1+rate].

        interp (string): Interpolation to use for re-sizing
        ('nearest', 'lanczos', 'bilinear', 'bicubic' or 'cubic').
    """

    def __init__(self, rate=0.0, interp='bilinear'):
        self.rate = rate

        self.interpolation = interp

    def __call__(self, clip):
        scaling_factor = random.uniform(1-self.rate, 1+self.rate)

        is_PIL = isinstance(clip[0], PIL.Image.Image)
        if is_PIL:
            clip = [np.asarray(img) for img in clip]

        im_h = clip[0].shape[0]
        im_w = clip[0].shape[1]
        new_w = int(im_w*scaling_factor)
        new_h = int(im_h*scaling_factor)
        data_final = []
        for image in clip:
            new_image = cv2.resize(image, (new_w, new_h), interpolation = self._get_cv2_interp(self.interpolation))
            if scaling_factor >= 1.0:
                top_x = (new_w-im_w) // 2
                top_y = (new_h-im_h) // 2
                cropped_image = new_image[top_y:top_y+im_h, top_x:top_x+im_w]
                data_final.append(cropped_image)
            else:
                border_w1 = (im_w-new_w) // 2
                border_w2 = im_w-new_w-border_w1
                border_h1 = (im_h-new_h) // 2
                border_h2 = im_h-new_h-border_h1
                cropped_image = cv2.copyMakeBorder(new_image, border_h1, border_h2, border_w1, border_w2, cv2.BORDER_CONSTANT, 0)
                data_final.append(cropped_image)

        if is_PIL:
            return [PIL.Image.fromarray(img) for img in data_final]
        else:
            return data_final

    def _get_cv2_interp(self, interp):
        if interp == 'nearest':
            return cv2.INTER_NEAREST
        elif interp == 'lanczos':
            return cv2.INTER_LANCZOS4
        elif interp == 'bilinear':
            return cv2.INTER_LINEAR
        elif interp == 'bicubic':
            return cv2.INTER_CUBIC
        elif interp == 'cubic':
            return cv2.INTER_CUBIC


class FixedResize(object):
    """
    Non-random resizing to a fixed size.

    Args:
        size (tuple): Desired output size in format (h, w).
    """

    def __init__(self, size):
        self.width = size[1]
        self.height = size[0]

    def __call__(self, clip):
        is_PIL = isinstance(clip[0], PIL.Image.Image)
        if is_PIL:
            clip = [np.asarray(img) for img in clip]

        results = []
        for image in clip:
            border_v = 0
            border_h = 0
            if (self.height / self.width) >= (image.shape[0] / image.shape[1]):
                border_v = int((((self.height / self.width) * image.shape[1]) - image.shape[0]) / 2)
            else:
                border_h = int((((self.width / self.height) * image.shape[0]) - image.shape[1]) / 2)

            new_image = cv2.copyMakeBorder(image, border_v, border_v, border_h, border_h, cv2.BORDER_CONSTANT, 0)
            new_image = cv2.resize(new_image, (self.width, self.height))
            results.append(new_image)

        if is_PIL:
            return [PIL.Image.fromarray(img) for img in results]
        else:
            return results


class RandomTranslate(object):
    """
      Shifting video in X and Y directions.

        Args:
            x_rate (float) : Translate in x direction [0-1], selected
            randomly from [-x_rate, +x_rate] pixels.

            y_rate (float) : Translate in y direction [0-1], selected
            randomly from [-y_rate, +y_rate] pixels.
    """

    def __init__(self, x_rate: float, y_rate: float):
        self.x_rate = x_rate
        self.y_rate = y_rate

    def __call__(self, clip):
        x_move = random.uniform(-self.x_rate, +self.x_rate)
        y_move = random.uniform(-self.y_rate, +self.y_rate)

        if isinstance(clip[0], np.ndarray):
            rows = clip[0].shape[0]
            cols = clip[0].shape[1]
            transform_mat = np.float32([[1, 0, x_move * cols], [0, 1, y_move * rows]])
            return [cv2.warpAffine(img, transform_mat, (cols, rows)) for img in clip]
        elif isinstance(clip[0], PIL.Image.Image):
            return [img.transform(img.size, PIL.Image.AFFINE, (1, 0, x_move * img.size[0], 0, 1, y_move * img.size[1])) for img in clip]
        else:
            raise TypeError('Expected numpy.ndarray or PIL.Image' +
                            'but got list of {0}'.format(type(clip[0])))


class RandomShear(object):
    """
    Shearing video in X and Y directions.

    Args:
        x_rate (float) : Shear rate in x direction [0-1], selected randomly from
        [0, +x_rate].

        y_rate (float) : Shear rate in y direction [0-1], selected randomly from
        [0, +y_rate].
    """

    def __init__(self, x_rate: float, y_rate: float):
        self.x_rate = x_rate
        self.y_rate = y_rate

    def __call__(self, clip):
        x_shear = random.uniform(0, self.x_rate)
        y_shear = random.uniform(0, self.y_rate)

        is_PIL = isinstance(clip[0], PIL.Image.Image)
        if is_PIL:
            clip = [np.asarray(img) for img in clip]

        im_h = clip[0].shape[0]
        im_w = clip[0].shape[1]
        transform_mat = np.float32([[1, x_shear, 0], [y_shear, 1, 0]])
        nW = int(clip[0].shape[1] + abs(x_shear*clip[0].shape[0]))
        nH = int(clip[0].shape[0]+abs(y_shear*clip[0].shape[1]))
        data_final = []
        for image in clip:
            new_image = cv2.warpAffine(image, transform_mat, (nW, nH))
            top_x = (nW-im_w) // 2
            top_y = (nH-im_h) // 2
            cropped_image = new_image[top_y:top_y+im_h, top_x:top_x+im_w]
            data_final.append(cropped_image)

        if is_PIL:
            return [PIL.Image.fromarray(img) for img in data_final]
        else:
            return data_final
