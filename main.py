from collections import OrderedDict
import numpy as np
import argparse
import dlib
import cv2
import os
from tqdm import tqdm

# arguments
ap = argparse.ArgumentParser()
ap.add_argument('-p', '--shape-predictor', default='./assets/shape_predictor_68_face_landmarks.dat',
                help='path to facial landmark predictor')
ap.add_argument('--image', default='data/test.jpg', help='path to input image')
ap.add_argument('--input_folder', default='data/face', help='the folder of images')
ap.add_argument('--output_folder', default='data/face_with_mask',
                help='the output folder you want to save images with mask')
ap.add_argument('--mode', default='mask',
                help='mask: input one image and output the image with mask, '
                     'mask_folder: input the folder path of images in order to apply masks to all the images'
                     'keyPoint: input an image and output the image with its key points and RoIs')
args = vars(ap.parse_args())


# OrderedDict
# face key points
FACIAL_LANDMARKS_68_IDXS = OrderedDict([
    ('mouth', (48, 68)),
    ('right_eyebrow', (17, 22)),
    ('left_eyebrow', (22, 27)),
    ('right_eye', (36, 42)),
    ('left_eye', (42, 48)),
    ('nose', (27, 36)),
    ('jaw', (0, 17))
])

FACIAL_LANDMARKS_5_IDXS = OrderedDict([
 ("right_eye", (2, 3)),
 ("left_eye", (0, 1)),
 ("nose", (4)),
])


def shape_to_np(shape, dtype='int'):
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords


def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
    # two copies: overlay and one for the final output image
    overlay = image.copy()
    output = image.copy()

    if colors is None:
        colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
                  (168, 100, 168), (158, 163, 32),
                  (163, 38, 32), (180, 42, 220)]

    for (i, name) in enumerate(FACIAL_LANDMARKS_68_IDXS.keys()):
        # coordinate of one point
        (j, k) = FACIAL_LANDMARKS_68_IDXS[name]
        pts = shape[j:k]
        if name == 'jaw':
            # connect points with line
            for l in range(1, len(pts)):
                ptA = tuple(pts[l-1])
                ptB = tuple(pts[l])
                cv2.line(overlay, ptA, ptB, colors[i], 2)

        else:

            # draw contours
            hull = cv2.convexHull(pts)
            cv2.drawContours(overlay, [hull], -1, colors[i], -1)

    cv2.addWeighted(overlay, alpha, output, 1-alpha, 0, output)
    return output


def add_mask(input_image_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args['shape_predictor'])
    image = cv2.imread(input_image_path)
    rects = detector(image, 1)
    for (i, rect) in enumerate(rects):
        shape = predictor(image, rect)
    points = shape
    rgbImg = image.copy()
    landmarks = [(p.x, p.y) for p in points.parts()]
    pts = np.array(
        [[landmarks[1], landmarks[2], landmarks[3], landmarks[4], landmarks[5], landmarks[6], landmarks[7],
          landmarks[8], landmarks[9], landmarks[10], landmarks[11], landmarks[12], landmarks[13], landmarks[14],
          landmarks[15], landmarks[29]]], dtype=np.int32)
    color = (np.random.uniform(0, 255), np.random.uniform(0, 255), np.random.uniform(0, 255))
    image_with_mask = cv2.fillPoly(rgbImg, pts, color, lineType=8)
    if image_with_mask is None:
        print(f"Image: {input_image_path} : Couldn't find a face to apply synthetic mask")
        return 0
    cv2.imwrite('{}_with_mask.png'.format(input_image_path), image_with_mask)
    cv2.imshow('image_with_mask', image_with_mask)
    cv2.waitKey(0)
    # visualize the key points used to generate mask
    # if you need it, just comment it and edit the path
    # keypoint = image.copy()
    # keypoint_mask = rgbImg.copy()
    # for (x, y) in pts[0]:
    #     cv2.circle(keypoint, (x, y), 3, (0, 0, 255), -1)
    #     cv2.circle(keypoint_mask, (x, y), 3, (0, 0, 255), -1)
    # if image_with_keypoint is None:
    #     print('error')
    # cv2.imshow('image with keypoint', image_with_keypoint)
    # cv2.waitKey(0)
    # cv2.imwrite('data/keypoint.png', keypoint)
    # cv2.imwrite('data/keypoint_mask.png', keypoint_mask)


#  convert all the images in input_dir_path to image_with_mask and save into output_dir_path
def add_mask_folder(input_dir_path, output_dir_path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args['shape_predictor'])
    if os.path.isdir(input_dir_path):
        if os.path.isdir(output_dir_path):
            os.mkdir(output_dir_path)

        for face_image_name in tqdm(os.listdir(input_dir_path)):
            face_image_path = f"{input_dir_path}/{face_image_name}"
            if not os.path.isfile(face_image_path):
                continue
            image = cv2.imread(face_image_path)
            rects = detector(image, 1)
            #rect = max(rects, key=lambda rect: rect.width() * rect.height())
            for (i, rect) in enumerate(rects):
                shape = predictor(image, rect)
            points = shape
            rgbImg = image.copy()
            landmarks = [(p.x, p.y) for p in points.parts()]
            pts = np.array(
                [[landmarks[1], landmarks[2], landmarks[3], landmarks[4], landmarks[5], landmarks[6], landmarks[7],
                  landmarks[8], landmarks[9], landmarks[10], landmarks[11], landmarks[12], landmarks[13], landmarks[14],
                  landmarks[15], landmarks[29]]], dtype=np.int32)
            color = (np.random.uniform(0, 255), np.random.uniform(0, 255), np.random.uniform(0, 255))
            image_with_mask = cv2.fillPoly(rgbImg, pts, color, lineType=8)
            if image_with_mask is None:
                print(f"Image: {face_image_path} : Couldn't find a face to apply synthetic mask")
                continue
            cv2.imwrite('output_dir_path/{}.png'.format(face_image_name), image_with_mask)

    else:
        print("Please check your input directory path")


# detect key points and visualize
def key_point_detection():

    # detector and predictor of dlib
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args['shape_predictor'])

    # read image
    image = cv2.imread(args['image'])

    # data pre-processing: you can uncomment it if you need
    # (h, w) = image.shape[:2]
    # width = 500
    # r = width / float(w)
    # dim = (width, int(r*h))
    # image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    # use gray image to increase the accuracy of face-detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # get bounding boxes
    rects = detector(gray, 1)

    # traverse each bounding box
    for (i, rect) in enumerate(rects):
        # key-point
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)

        for (name, (i, j)) in FACIAL_LANDMARKS_68_IDXS.items():
            clone = image.copy()
            cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
            # plot key point
            for (x, y) in shape[i:j]:
                cv2.circle(clone, (x, y), 3, (0, 0, 255), -1)

            (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))

            # region of interest
            roi = image[y:y+h, x:x+w]
            (h, w) = roi.shape[:2]
            width = 250
            r = width / float(w)
            dim = (width, int(r*h))
            roi = cv2.resize(roi, dim, interpolation=cv2.INTER_AREA)

            # show
            cv2.imshow('ROI', roi)
            cv2.imshow('Image', clone)
            cv2.waitKey(0)

        output = visualize_facial_landmarks(image, shape)
        cv2.imshow('Image', output)
        cv2.waitKey(0)


if __name__ == '__main__':
    if args['mode'] == 'mask_folder':
        add_mask_folder(args['input_folder'], args['output_folder'])
    elif args['mode'] == 'mask':
        add_mask(args['image'])
    elif args['mode'] == 'keyPoint':
        key_point_detection()
