import cv2
import my_bag_utils as bu

def crop_image(im, expected_shape, new_shape):
    assert (im.shape == expected_shape), 'im.shape: {}, expected_shape: {}'.format(im.shape, expected_shape)

    cropped = im[:new_shape[0], :new_shape[1]]

    assert(cropped.shape == new_shape)

    return cropped

def _crop_images(image_paths, expected_shape, new_shape):
    count = 0
    for path in image_paths:
        im = cv2.imread(path)
        cropped = crop_image(im, expected_shape, new_shape)
        cv2.imwrite(path, cropped)
        count += 1
        if count % 1000 == 0:
            print('Cropped {} out of {} images.'.format(count, len(image_paths)))

def crop_images(dir, pattern, expected_shape, new_shape):
    image_paths = bu.find_files(dir, pattern)
    print('Found {} images.'.format(len(image_paths)))
    _crop_images(image_paths, expected_shape, new_shape)

if __name__ == '__main__':
    crop_images('/home/eljefec/repo/squeezeDet/data/KITTI/training/', '*.png', (801, 801, 3), (800, 800, 3))
