import os
os.environ["KERAS_BACKEND"] = "torch"  # @param ["tensorflow", "jax", "torch"]
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  #disable custom operations

import keras
import keras_cv
from keras_cv import bounding_box, visualization
from keras.metrics import Precision, Recall
import numpy as np
import tqdm
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import time

def checkIfInRegion(centerPointX, centerPointY):
    # [x1, y1, x2, y2]
    # (x1, y1)---
    # |         |
    # |---------(x2, y2)
    northRegion = [820, 460, 920, 560]
    westRegion = [210, 600, 335, 710]
    south1Region = [1060, 890, 1230, 1070]
    south2Region = [790, 920, 990, 1080]
    eastRegion = [1580, 620, 1660, 720]
    inRegion = False
    region = None

    # check if in north region
    if ((centerPointX >= northRegion[0]) and (centerPointX <= northRegion[2])):
        if ((centerPointY >= northRegion[1]) and (centerPointY <= northRegion[3])):
            inRegion = True
            region = "north"

    # check if in west region
    if ((centerPointX >= westRegion[0]) and (centerPointX <= westRegion[2])):
        if ((centerPointY >= westRegion[1]) and (centerPointY <= westRegion[3])):
            inRegion = True
            region = "west"

    #check if in south1 region
    if ((centerPointX >= south1Region[0]) and (centerPointX <= south1Region[2])):
        if ((centerPointY >= south1Region[1]) and (centerPointY <= south1Region[3])):
            inRegion = True
            region = "south1"

    #check if in south2 region
    if ((centerPointX >= south2Region[0]) and (centerPointX <= south2Region[2])):
        if ((centerPointY >= south2Region[1]) and (centerPointY <= south2Region[3])):
            inRegion = True
            region = "south2"

    #check if in east region
    if ((centerPointX >= eastRegion[0]) and (centerPointX <= eastRegion[2])):
        if ((centerPointY >= eastRegion[1]) and (centerPointY <= eastRegion[3])):
            inRegion = True
            region = "east"

    return region

def getInt(printMessage, needPos = False, needGreaterThanZero = False):
    gotGoodVal = False
    output = 0
    while not gotGoodVal:
        try:
            output = input(printMessage)
            print()
            output = int(output)
            if needPos:
                if needGreaterThanZero:
                    if output > 0:
                        gotGoodVal = True
                    else:
                        print("Invalid entry, entry must be positive.")
                else:
                    gotGoodVal = True
            else:
                gotGoodVal = True
        except KeyboardInterrupt:
            exit()
        except:
            print("Invalid entry, entry must be an integer.")
    return output

def getChar(printMessage, acceptedChars = None):
    gotGoodVal = False
    output = ''
    if acceptedChars == None:
        print("Error: no provided char list")

    while not gotGoodVal:
        try:
            output = input(printMessage)
            print()
            if output in acceptedChars:
                gotGoodVal = True
        except KeyboardInterrupt:
            exit()
        except:
            print("Invalid entry, entry must be an integer.")
    return output

outputPath = "output"
croppedOut = "final"

pretrained_model = keras_cv.models.YOLOV8Detector.from_preset(
    "yolo_v8_m_pascalvoc", bounding_box_format="xywh"
)

class_ids = [
    "bus",
    "car",
    "pickup",
    "suv"
]
class_mapping = dict(zip(range(len(class_ids)), class_ids))

tempChar = getChar("Train the model? (Doesn't work right, so hit 'N'.", ['y', 'Y', 'n', 'N'])
if tempChar == 'y' or tempChar == 'Y':
    train = True
else:
    train = False

if train:
    pretrained_model = keras_cv.models.ImageClassifier.from_preset(
        "mobilenet_v3_small", num_classes=4
    )

    trainSet = keras.utils.image_dataset_from_directory(
        directory='training_data/',
        labels='inferred',
        label_mode='categorical',
        batch_size=23,
        image_size=(256, 256))

    validationSet = keras.utils.image_dataset_from_directory(
        directory='validation_data/',
        labels='inferred',
        label_mode='categorical',
        batch_size=6,
        image_size=(256, 256))

    pretrained_model.compile(
        loss="mean_squared_error",
        optimizer=keras.optimizers.SGD(learning_rate=0.01),
        metrics=["accuracy", Precision(), Recall()],
        )

    pretrained_model.fit(trainSet)
    #pretrained_model.predict(validationSet)
    #testResults = pretrained_model.evaluate(validationSet)
    loss, accuracy, precision, recall = pretrained_model.evaluate(validationSet)
    f1Score = 0
    #testResults = pretrained_model.predict(validationSet)
    #f1Score = ((precision * recall)/(precision + recall))
    #print(trainResults)
    print(f"Loss: {loss}, Accuracy: {accuracy}, F1: {f1Score}, Precision: {precision}, Recall: {recall}")
    tempChar = getChar("Please note: This trained model is not suitable yet.\nContinue with it? 'Y' or 'N'", ['y', 'Y', 'n', 'N'])
    if tempChar == 'y' or tempChar == 'Y':
        train = True
    else:
        train = False
    if not tempChar:
        pretrained_model = keras_cv.models.YOLOV8Detector.from_preset(
            "yolo_v8_m_pascalvoc", bounding_box_format="xywh"
            )

gotVideo = False
try:
    while not gotVideo:
        videoName = input("Video file: ")
        gotVideo = os.path.isfile(videoName)
        if not gotVideo:
            print("Invalid file. Make sure the file is in the same directory as this python file.")
except KeyboardInterrupt:
    exit()

framesPerSecond = getInt("Video frames per second: ", True, True)
intervalLengthSeconds = getInt("Interval length in seconds: ", True, True)
intervals = getInt("How many intervals: ", True, True)
intervalCount = getInt("How many intervals to skip: \n(if you are trying to resume a failed analysis enter the last interval mentioned, minus one.\nOtherwise, just hit 0.)", True)
startTimeHour = getInt("Start time hour: ", True, False)
startTimeMin = getInt("Start time min: ", True, False)
letterDay = input("Letter day: ")

if intervalCount != 0:
    i = 0
    while i < (intervalCount):
        startTimeMin += 1
        if startTimeMin > 59:
            startTimeHour += 1
            startTimeMin = 0
        i += 1
    print(f"Skipping to start at ", end='')
    print(f"{'{:02d}'.format(startTimeHour)}:{'{:02d}'.format(startTimeMin)}")

tempChar = getChar("Print initial column headers?", ['y', 'Y', 'n', 'N'])
if tempChar == 'y' or tempChar == 'Y':
    printHeading = True
else:
    printHeading = False

if (printHeading):
    f = open("output.txt", "a")
    f.write(f"Day,Time Start,Time End,North,South,East,West\n")
    f.close()

tempChar = getChar("Save analyzer output figures?", ['y', 'Y', 'n', 'N'])
if tempChar == 'y' or tempChar == 'Y':
    saveFig = True
else:
    saveFig = False

capture = cv2.VideoCapture(videoName)

maxFrames = (framesPerSecond * intervalLengthSeconds) * intervals
counterNorth = {}
counterWest = {}
counterEast = {}
counterSouth = {}

start = time.process_time()

# make the output path
if saveFig:
    if not (os.path.exists(outputPath)):
        os.makedirs(outputPath)

bounding_boxes = {
  "classes": [0], # 0 is an arbitrary class ID
  "boxes": [[0.25, 0.4, .15, .1]]
   # bounding box is in "rel_xywh" format
   # so 0.25 represents the start of the bounding box 25% of
   # the way across the image.
   # The .15 represents that the width is 15% of the image width.
}

cleanups = 0
framesSinceLastInterval = 0
carsWentNorth = 0
carsWentEast = 0
carsWentWest = 0
carsWentSouth = 0
wasInNorth = False
wasInEast = False
wasInWest = False
wasInSouth1 = False
wasInSouth2 = False
currHour = startTimeHour
currMin = startTimeMin
overallFrame = 0
startFrame = intervalCount * (framesPerSecond * intervalLengthSeconds)

while intervalCount < intervals:
    images=[]
    count = 0
    processed = 0
    prevHour = currHour
    prevMin = currMin
    currMin += 1
    if currMin > 59:
        currHour += 1
        currMin = 0

    print(f"Interval {intervalCount + 1} of {intervals}")

    i = 0
    while i < (intervalLengthSeconds * framesPerSecond):
        grabFrame = (startFrame + overallFrame + i) - 1
        print(f"Frame {i+1} of {(framesPerSecond * intervalLengthSeconds)} - interval {(intervalCount + 1)}")
        capture.set(cv2.CAP_PROP_POS_FRAMES, grabFrame)
        success, image = capture.read()
        inference_resizing = keras_cv.layers.Resizing(
            960, 960, pad_to_aspect_ratio=True, bounding_box_format="xywh"
        )
        image = np.array([image])

        #image_batch = inference_resizing([image])
        image_batch = inference_resizing(image)

        y_pred = pretrained_model.predict(image_batch)
        # y_pred is a bounding box Tensor:
        # {"classes": ..., boxes": ...}

        boxes = y_pred["boxes"]
        region = None

        carInNorth = False
        carInEast = False
        carInWest = False
        carInSouth1 = False
        carInSouth2 = False
        for box in boxes[0]:
            if box[0] != -1:
                # box coordinate: [x, y, width, height]
                centerPointX = (box[0] + (box[2] / 2)) * 2
                centerPointY = (box[1] + (box[3] / 2)) * 2

                #print(box)

                region = checkIfInRegion(centerPointX, centerPointY)

                if region == "north":
                    if not wasInNorth:
                        print("Car entered North")
                    carInNorth = True
                elif region == "west":
                    if not wasInWest:
                        print("Car entered West")
                    carInWest = True
                elif region == "east":
                    if not wasInEast:
                        print("Car entered East")
                    carInEast = True
                elif region == "south1":
                    if not wasInSouth1:
                        print("Car entered south1")
                    carInSouth1 = True
                elif region == "south2":
                    if not wasInSouth2:
                        print("Car entered south2")
                    carInSouth2 = True

        if not carInNorth and wasInNorth:
            wasInNorth = False
            print("Car left North")
            carsWentNorth += 1

        if not carInWest and wasInWest:
            wasInWest = False
            print("Car left West")
            carsWentWest += 1

        if not carInEast and wasInEast:
            wasInEast = False
            print("Car left West")
            carsWentEast += 1

        if not carInSouth1 and wasInSouth1:
            wasInSouth1 = False
            print("Car left South1")
            carsWentSouth += 1

        if not carInSouth2 and wasInSouth2:
            wasInSouth2 = False
            print("Car left South2")
            carsWentSouth += 1

        wasInNorth = carInNorth
        wasInEast = carInEast
        wasInSouth1 = carInSouth1
        wasInSouth2 = carInSouth2
        wasInWest = carInWest

        if saveFig:
            visualization.plot_bounding_box_gallery(
                image_batch,
                value_range=(0, 255),
                rows=1,
                cols=1,
                y_pred=y_pred,
                scale=19.2,
                font_scale=0.7,
                bounding_box_format="xywh",
                class_mapping=class_mapping,
            )

            outputString = outputPath + "//" + "frame-{0}.png".format(overallFrame)
            plt.savefig(outputString)

        if i % 19 == 0:
            plt.close('all')

        overallFrame += 1
        i += 1

    counterNorth[f"{intervalCount}"] = carsWentNorth
    counterWest[f"{intervalCount}"] = carsWentWest
    counterEast[f"{intervalCount}"] = carsWentEast
    counterSouth[f"{intervalCount}"] = carsWentSouth

    carsWentNorth = 0
    carsWentEast = 0
    carsWentWest = 0
    carsWentSouth = 0
    print('-' * 40)
    print(f"Interval: {intervalCount}")
    print(f"North: {counterNorth[f'{intervalCount}']}")
    print(f"South: {counterSouth[f'{intervalCount}']}")
    print(f"East: {counterEast[f'{intervalCount}']}")
    print(f"West: {counterWest[f'{intervalCount}']}")
    print()

    f = open("output.txt", "a")
    #f.write(f"Interval: {intervalCount + 1}\n")
    #f.write(f"North: {counterNorth[f'{intervalCount}']}\n")
    #f.write(f"South: {counterSouth[f'{intervalCount}']}\n")
    #f.write(f"East: {counterEast[f'{intervalCount}']}\n")
    #f.write(f"West: {counterWest[f'{intervalCount}']}\n")
    #f.write('\n\n')
    f.write(f"{letterDay},{'{:02d}'.format(prevHour)}:{'{:02d}'.format(prevMin)},")
    f.write(f"{'{:02d}'.format(currHour)}:{'{:02d}'.format(currMin)},")
    f.write(f"{counterNorth[f'{intervalCount}']},{counterSouth[f'{intervalCount}']},")
    f.write(f"{counterEast[f'{intervalCount}']},{counterWest[f'{intervalCount}']}\n")
    f.close()

    intervalCount += 1

print("Analysis complete: Time elapsed: ")
print(time.process_time() - start)