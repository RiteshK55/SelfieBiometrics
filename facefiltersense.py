import cv2
from keras.models import load_model
import time
import numpy as np
import pandas as pd


def loadAllModels():
    """
    This function loads all the saved models

    Returns:
        Loaded saved models which are ready to be infered
    """
    tic = time.time()
    face_model = load_model('weights/action.h5', compile=False)
    toc = time.time()
    print('Time taken to load Face model: ', toc - tic, ' seconds')

    tic = time.time()
    gender_model = load_model('weights/action_gender.h5', compile=False)
    toc = time.time()
    print('Time taken to load Gender model: ', toc - tic, ' seconds')

    tic = time.time()
    age_model = load_model('weights/action_age.h5', compile=False)
    toc = time.time()
    print('Time taken to load Age model: ', toc - tic, ' seconds')

    tic = time.time()
    eth_model = load_model('weights/action_eth.h5', compile=False)
    toc = time.time()
    print('Time taken to load Ethnicity model: ', toc - tic, ' seconds')

    return face_model, gender_model, age_model, eth_model


face_model, gender_model, age_model, eth_model = loadAllModels()

# Defining list of labels
persons = [str(num) for num in range(1, 103)]
persons.sort()
genders = ['female', 'male']
eths = ['black', 'east_asian', 'west_asian', 'white']

# Setting the image dimensions for model input
img_height, img_width = 256, 256


def find(file_path):
    """
    This function predicts the best possible match of a face in the given dataset
    Args:
        file_path (str): file path to the image file to be preprocessed

    Returns:
        df:  pandas dataframe of predicted labels sorted in the order of probability
    """
    # Reading the input image and resizing to model input dimensions
    image = cv2.imread(file_path)
    image_resized = cv2.resize(image, (img_height, img_width))
    image = np.expand_dims(image_resized, axis=0)

    # Calculating processing time
    tic = time.time()
    pred = face_model.predict(image)[0]  # model inference
    # list of predicted labels sorted by probability
    identity_ids = np.flip(np.argsort(pred))
    # creating a dataframe to store predicted label with probabilities in sorted order
    data = []
    for i in identity_ids:
        data.append([persons[i], pred[i]])
    toc = time.time()

    processing_time = toc - tic
    print("Time taken = {} seconds".format(processing_time))
    columns = ['Predicted Label', 'Probability']
    df = pd.DataFrame(data=np.array(data), columns=columns)

    print("Predicted Label is: {} with probability = {}".format(
        df.iloc[0]['Predicted Label'], df.iloc[0]['Probability']))

    # returns the dataframe of possible predicted labels sorted in the order of probability
    return df


def findEuclideanDistance(source_representation, test_representation):
    """
    This function calculates the Euclidean distance between two given vectors
    Args:
        source_representation: vector representation of first face
        test_representation: vector representation of second face

    Returns:
        euclidean_distance: numpy array giving euclidean distance between the
                            given vector represntations
    """
    # if the input type is list, convert to numpy array
    if type(source_representation) == list:
        source_representation = np.array(source_representation)

    if type(test_representation) == list:
        test_representation = np.array(test_representation)

    # Calculating euclidean distance using formula
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(
        euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


def l2_normalize(vec):
    """
    This function calculates the L2-Norm of a given vector
    Args:
        vec: vector embedding

    Returns:
        l2_norm: numpy array representing L2 Normalized form of given vector
    """
    l2_norm = vec / np.sqrt(np.sum(np.multiply(vec, vec)))
    return l2_norm


def verify(file_path1, file_path2):
    """
    This function verifies if the given two images belong to the same person or not
    Args:
        file_path1 (str): file path to the first face image
        file_path2 (str): file path to the second face image        

    Returns:
        result: dictionary with two keys:
                'verified': bool value, true if faces are same else false
                'distance': distance value between input faces
    """

    # Reading the input images and resizing to model input dimensions
    image1 = cv2.imread(file_path1)
    image_resized1 = cv2.resize(image1, (img_height, img_width))
    image1 = np.expand_dims(image_resized1, axis=0)

    image2 = cv2.imread(file_path2)
    image_resized2 = cv2.resize(image2, (img_height, img_width))
    image2 = np.expand_dims(image_resized2, axis=0)

    # Calculating processing time
    tic = time.time()
    # model inference
    pred1 = face_model.predict(image1)
    pred2 = face_model.predict(image2)
    toc = time.time()

    # Calculating L2 normalized Euclidean distance between faces
    distance = findEuclideanDistance(l2_normalize(pred1), l2_normalize(pred2))

    # Initializing verified to default value of False
    verified = False

    # if distance between faces is less than threshold, then verified is True
    if distance < 0.75:
        verified = True

    processing_time = toc - tic
    print("Time taken = {} seconds".format(processing_time))
    result = {'verified': verified, 'distance': distance}
    return result


def analyze(file_path):
    """
    This function predicts the face identity, gender, age and ethnicity of the input face
    Args:
        file_path (str): file path to the image file to be preprocessed

    Returns:
        result: dictionary with the following keys:
                  'predicted_identity': Predicted face ID of the person,
                  'identity_probabilities': List of probabilities of best 5 matches,
                  'predicted_age': Predicted age of the person,
                  'predicted_gender': Predicted gender of the person,
                  'gender_probabilities': List of probabilities of all gender classes,
                  'predicted_ethnicity': Predicted ethnicity of the person,
                  'ethnicity_probabilities': List of probabilities of all ethnicity classes,
    """
    # Reading the input image and resizing to model input dimensions
    image = cv2.imread(file_path)
    image_resized = cv2.resize(image, (img_height, img_width))
    image = np.expand_dims(image_resized, axis=0)

    # Calculating processing time
    tic = time.time()
    pred = face_model.predict(image)[0]  # face model inference
    predicted_identity = persons[np.argmax(pred)]  # label with highest prob

    # Getting top 5 results
    identity_ids = np.flip(np.argsort(pred))
    identity = {}
    topK = 5
    for i in identity_ids:
        identity[persons[i]] = pred[i]
        topK -= 1
        if topK == 0:
            break
    
    baseline_img = "Dataset/Face Dataset/{}/{}.jpg".format(predicted_identity, predicted_identity)
    filtered_img = file_path
    
    verified_dict = verify(baseline_img, filtered_img)
    distance = verified_dict['distance']
    
    comments = ''
    if distance < 0.75:
        comments = 'The Filter DOES NOT produce significant distortion from the Baseline Image'
    else:
        comments = 'The Filter produces significant distortion from the Baseline Image'

    pred = gender_model.predict(image)[0]  # gender model inference
    # gender class with highest prob
    predicted_gender = genders[np.argmax(pred)]
    gender_ids = np.flip(np.argsort(pred))

    # Getting prob for each gender class
    gender = {}
    for i in gender_ids:
        gender[genders[i]] = pred[i]

    pred = eth_model.predict(image)[0]  # ethnicity model inference
    predicted_eth = eths[np.argmax(pred)]  # ethncity class with highest prob
    eth_ids = np.flip(np.argsort(pred))

    # Getting prob for each ethnicity class
    ethnicity = {}
    for i in eth_ids:
        ethnicity[eths[i]] = pred[i]

    pred = age_model.predict(image)  # age model inference
    age = round(pred[0][0])  # estimating the age to nearest integer

    toc = time.time()
    processing_time = toc - tic
    result = {'predicted_identity': predicted_identity,
              'comments on filter usability': comments,
              'identity_probabilities': identity,
              'predicted_age': age,
              'predicted_gender': predicted_gender,
              'gender_probabilities': gender,
              'predicted_ethnicity': predicted_eth,
              'ethnicity_probabilities': ethnicity
              }
    return result
