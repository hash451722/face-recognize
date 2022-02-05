import pathlib
import uuid

import cv2
import numpy as np


# (B, G, R)
BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
PURPLE = (255, 0, 255)


class Face():
    def __init__(self):
        self.root_dir = pathlib.Path(__file__).parent

        model_path = self.root_dir.joinpath("face_detection_yunet_2021dec.onnx")
        self.detector = cv2.FaceDetectorYN.create(
            model=str(model_path),
            config='',
            input_size=(320, 180)
        )

        model_path = self.root_dir.joinpath("face_recognition_sface_2021dec.onnx")
        self.recognizer = cv2.FaceRecognizerSF.create(str(model_path), "")

        self._cosine = 0  # cosine similarity
        self._norml2 = 1  # Norm-L2 distance
        self._threshold_cosine = 0.363
        self._threshold_norml2 = 1.128

    def detect(self, img_path=None):
        if img_path is None:
            img_path = self.root_dir.joinpath("sample.jpg")

        img = cv2.imread(str(img_path))
        height, width, _ = img.shape

        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(img)

        img_detected = self._visualize(img, faces)
        
        return img, faces, img_detected

    def crop(self, img, faces):
        cropped_faces = []
        for face in faces:
            face = list(map(int, face))
            cropped_face = img[face[1]:face[1]+face[3], face[0]:face[0]+face[2]]
            cropped_faces.append(cropped_face)
        return cropped_faces

    def align(self, img, faces):
        aligned_faces = []
        for face in faces:
            aligned_face = self.recognizer.alignCrop(img, face)
            aligned_faces.append(aligned_face)
            # cv2.imwrite(str(self.root_dir.joinpath("aligned_face" + str(idx) + ".jpg")), aligned_face)
            # print(aligned_face.shape)
        return aligned_faces
    
    def feature(self, aligned_faces):
        '''
        Extraction of face image features
        '''
        feature_list = []
        for face in aligned_faces:
            face_feature = self.recognizer.feature(face)
            feature_list.append(face_feature)
        return feature_list

    def match(self, feature1, feature2):
        score_cosine = self.recognizer.match(feature1, feature2, self._cosine)
        score_norml2 = self.recognizer.match(feature1, feature2, self._norml2)
        return score_cosine, score_norml2

    def save_image(self, img, file_path=None):
        if file_path is None:
            path = self.root_dir.joinpath("temp.jpg")
        cv2.imwrite(str(path), img)

    def convert_color(img):
        """
        BGR -> RGB
        """
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _visualize(self, img, faces):
        output = img.copy()

        for face in faces:
            coords = face[:-1].astype(np.int32)
            # face bounding box
            cv2.rectangle(output, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), GREEN, 2)
            # landmarks
            cv2.circle(output, (coords[4], coords[5]), 2, RED, 2)  # right eye
            cv2.circle(output, (coords[6], coords[7]), 2, BLUE, 2)  # left eye
            cv2.circle(output, (coords[8], coords[9]), 2, GREEN, 2)  # nose
            cv2.circle(output, (coords[10], coords[11]), 2, PURPLE, 2)  # right corner of the mouth
            cv2.circle(output, (coords[12], coords[13]), 2, YELLOW, 2)  # left corner of the mouth
            # score
            cv2.putText(output, '{:.3f}'.format(face[-1]), (coords[0], coords[1]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN)

        return output

    def save_img_feature(self, img, feature, file_path):
        '''
        extension: npz
        '''
        np.savez_compressed(file_path, img=img, feature=feature)
        

    def load_img_feature(self, file_path):
        '''
        extension: npz
        '''
        npz = np.load(file_path)

        return npz["img"], npz["feature"]


    def img2features(self, img_path, save_dir_path):
        '''
        Extracting and saving features from a single image
        '''
        if not pathlib.Path.is_dir(save_dir_path):
            raise Exception('Not directory')

        img, faces, _ = self.detect(img_path)
        aligned_faces = self.align(img, faces)
        features = self.feature(aligned_faces)

        for img, feature in zip(aligned_faces, features):
            file_name =str(uuid.uuid4()) + ".npz"
            npz_path = save_dir_path.joinpath(file_name)
            self.save_img_feature(img, feature, npz_path)



if __name__ == '__main__':

    face = Face()


    root_dir = pathlib.Path(__file__).parent
    img_path = root_dir.joinpath("sample.jpg")

    save_dir = root_dir
    face.img2features(img_path, save_dir)


    npz_files = root_dir.glob('**/*.npz')
    # print(list(npz_files))

    for npz in npz_files:
        print(type(npz))
        print(npz)


    # cropped_faces = face.crop(img, faces)

    # print(type(aligned_faces[0]))
    # print(type(features[0]))

    # npz_path = root_dir.joinpath("np_savez_comp.npz")

    # face.save_img_feature(aligned_faces[0], features[0], npz_path)

    # img, feature = face.load_img_feature(npz_path)
    # print(img.shape)
    # print(feature.shape)
