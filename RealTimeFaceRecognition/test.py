import warnings

warnings.filterwarnings("ignore")
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import cv2
from PIL import Image
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
from sklearn.preprocessing import Normalizer, LabelEncoder
import pickle
from tensorflow.keras.models import load_model


class FaceDetector:
    def __init__(self):
        self.facenet_model = load_model("facenet_keras.h5")
        self.svm_model = pickle.load(open("SVM_classifier.sav", 'rb'))
        self.data = np.load('faces_dataset_embeddings.npz')
        # object to the MTCNN detector class
        self.detector = MTCNN()
        self.model_mask = load_model('Mask/mask_detector.model', compile=True)
        self.prototxt = 'Mask/face_detector/deploy.prototxt'
        self.caffe_path = 'Mask/face_detector/res10_300x300_ssd_iter_140000.caffemodel'
        self.faceNet = cv2.dnn.readNet(self.prototxt, self.caffe_path)

    def face_localizer(self, person):
        """Method takes the extracted faces and returns the coordinates"""
        # 1. Get the coordinates of the face
        bounding_box = person['box']
        x1, y1 = abs(bounding_box[0]), abs(bounding_box[1])
        width, height = bounding_box[2], bounding_box[3]
        x2, y2 = x1 + width, y1 + height
        return x1, y1, x2, y2, width, height

    def face_preprocessor(self, frame, x1, y1, x2, y2, required_size=(160, 160)):
        """Method takes in frame, face coordinates and returns preprocessed image"""
        # 1. extract the face pixels
        face = frame[y1:y2, x1:x2]
        # 2. resize pixels to the model size
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = np.asarray(image)
        # 3. scale pixel values
        face_pixels = face_array.astype('float32')
        # 4. standardize pixel values across channels (global)
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        # 5. transform face into one sample
        samples = np.expand_dims(face_pixels, axis=0)
        # 6. get face embedding
        yhat = self.facenet_model.predict(samples)
        face_embedded = yhat[0]
        # 7. normalize input vectors
        in_encoder = Normalizer(norm='l2')
        X = in_encoder.transform(face_embedded.reshape(1, -1))
        return X

    def face_svm_classifier(self, X):
        """Methods takes in preprocessed images ,classifies and returns predicted Class label and probability"""
        # predict
        yhat = self.svm_model.predict(X)
        label = yhat[0]
        yhat_prob = self.svm_model.predict_proba(X)
        probability = round(yhat_prob[0][label], 2)
        trainy = self.data['arr_1']
        # predicted label decoder
        out_encoder = LabelEncoder()
        out_encoder.fit(trainy)
        predicted_class_label = out_encoder.inverse_transform(yhat)
        label = predicted_class_label[0]
        return label, str(probability)

    def mask_face(self, frame, labels):
        blobShape = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], True, False)
        self.faceNet.setInput(blobShape)
        detections = self.faceNet.forward()
        height = frame.shape[0]
        width = frame.shape[1]
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:  # face confidence
                x1 = int(detections[0, 0, i, 3] * width)
                y1 = int(detections[0, 0, i, 4] * height)
                x2 = int(detections[0, 0, i, 5] * width)
                y2 = int(detections[0, 0, i, 6] * height)

                try:
                    img_face = frame[y1:y2, x1:x2]

                    img_resize = cv2.resize(img_face, dsize=(224, 224),
                                            interpolation=cv2.INTER_CUBIC)
                    img_resize = cv2.normalize(img_resize, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,
                                               dtype=cv2.CV_32F)

                    img_1 = img_resize[np.newaxis, :]
                    prediction = self.model_mask.predict(img_1)
                    predicted_index = np.argmax(prediction, axis=1)
                    predicted_proba = np.max(prediction)
                    # print(predicted_index)

                    text = labels[predicted_index[0].item()] + " " + \
                           str(round(predicted_proba, 2))
                except:
                    break
        return predicted_index, text

    def face_detector(self):
        labels = ["with_mask", "without_mask"]
        cap = cv2.VideoCapture(0)
        while True:
            _, frame = cap.read()
            result = self.detector.detect_faces(frame)
            for person in result:
                x1, y1, x2, y2, width, height = self.face_detector(person)
                X = self.face_preprocessor(frame, x1, y1, x2, y2, required_size=(160, 160))
                label_face, probability = self.face_svm_classifier(X)
            if float(probability) > 0.7:
                print(" Person : {} , Probability : {}".format(label_face, probability))
                predicted_index, text = FaceDetector.mask_face(frame, labels)
                if labels[predicted_index[0].item()] == "with_mask":
                    cv2.putText(frame, text, (x1 - 15, y1 - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                elif labels[predicted_index[0].item()] == "without_mask":
                    cv2.putText(frame, text, (x1 - 15, y1 - 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                print("Unknow")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 155, 255), 2)
            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    facedetector = FaceDetector()
    facedetector.face_detector()
