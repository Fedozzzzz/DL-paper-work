import os
from imutils import paths
import pickle
from imutils.face_utils import FaceAligner
import imutils
import dlib
import cv2


def extract_embeddings(img_dir, embeddings_path):
    # load our serialized face embedding model from disk
    print("[INFO] loading face recognizer...")
    embedder = cv2.dnn.readNetFromTorch('openface_nn4.small2.v1.t7')
    detector = dlib.get_frontal_face_detector()
    print("[INFO] loading face predictor for image preprocessing...")
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')
    fa = FaceAligner(predictor, desiredFaceWidth=256)

    print("[INFO] quantifying faces...")
    image_paths = list(paths.list_images('celeb_dataset'))[0:10400]
    image_paths = list(paths.list_images(img_dir))

    known_embeddings = []
    known_names = []

    total_faces_processed = 0

    # loop over the image paths
    for (i, image_path) in enumerate(image_paths):
        # extract the person name from the image path
        print("[INFO] processing image {}/{}".format(i + 1,
                                                     len(image_paths)))
        name = image_path.split(os.path.sep)[-2]
        print("name:", name)
        # load the image, resize it to have a width of 600 pixels (while
        # maintaining the aspect ratio), and then grab the image
        # dimensions
        image = cv2.imread(image_path)
        if image is None:
            continue

        image = imutils.resize(image, width=600)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detections = detector(gray, 2)

        # ensure at least one face was found
        if len(detections) > 0:
            for detection in detections:
                # (x, y, w, h) = rect_to_bb(detection)
                # extract the face ROI and grab the ROI dimensions
                # face = image[startY:endY, startX:endX]
                # faceOrig = imutils.resize(image[y:y + h, x:x + w], width=256)
                faceAligned = fa.align(image, gray, detection)
                # display the output images
                # cv2.imshow("Original", faceOrig)
                # cv2.imshow("Aligned", faceAligned)
                # cv2.waitKey(0)
                (fH, fW) = faceAligned.shape[:2]
                # ensure the face width and height are sufficiently large
                if fW < 20 or fH < 20:
                    continue
                # construct a blob for the face ROI, then pass the blob
                # through our face embedding model to obtain the 128-d
                # quantification of the face
                faceBlob = cv2.dnn.blobFromImage(faceAligned, 1.0 / 255,
                                                 (96, 96), (0, 0, 0), swapRB=True, crop=False)
                embedder.setInput(faceBlob)
                vec = embedder.forward()
                # add the name of the person + corresponding face
                # embedding to their respective lists
                known_names.append(name)
                known_embeddings.append(vec.flatten())
                total_faces_processed += 1

    # dump the facial embeddings + names to disk
    print("[INFO] serializing {} encodings...".format(total_faces_processed))
    data = {"embeddings": known_embeddings, "names": known_names}
    f = open(embeddings_path, "wb")
    f.write(pickle.dumps(data))
    f.close()


if __name__ == '__main__':
    ds_path = 'celeb_dataset'
    embeddings_path = 'output/embeddings.pickle'
    extract_embeddings(ds_path, embeddings_path)
