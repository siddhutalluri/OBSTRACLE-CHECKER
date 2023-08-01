import cv2
import tensorflow as tf
import numpy as np
import time

model = tf.keras.models.load_model('mymodel.h5')

IMAGE_SIZE = [128, 128]
IMAGE_SHAPE = IMAGE_SIZE + [3,]


from collections import namedtuple
Label = namedtuple( 'Label' , [
    'name'        , 'id'          , 'trainId'     , 'category'    ,
    'categoryId'  , 'hasInstances', 'ignoreInEval', 'color'
    ] )

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , ( 0,  0,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 0,  0,  0) ),
    Label(  'road'                 ,  7 ,        0 , 'ground'          , 1       , False        , False        , (255,255,255) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'ground'          , 1       , False        , False        , (0,0,0) ),
    Label(  'parking'              ,  9 ,      255 , 'ground'          , 1       , False        , True         , ( 0,  0,  0) ),
    Label(  'rail track'           , 10 ,      255 , 'ground'          , 1       , False        , True         , (0,  0,  0) ),
    Label(  'building'             , 11 ,        2 , 'constructiox'
                                                     'n'    , 2       , False        , False        , ( 0,  0,  0) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (0,  0,  0) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (0,0,0) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (0,  0,  0) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (0,  0,  0) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (0,  0,  0) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (0,  0,  0) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (0,  0,  0) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (0,  0,  0) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (0,  0,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (0,  0,  0) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (0,  0,  0) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , (0,  0,  0) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , ( 0,  0,  0) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , ( 0,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , ( 0,  0,  0) ),

Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , ( 0,  0,  0) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , ( 0,  0,  0) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , ( 0,  0,  0) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , ( 0,  0,  0) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , ( 0,  0,  0) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,0) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (0,  0,0) ),
    Label(  'license plate'        , 34 ,       19 , 'vehicle'         , 7       , False        , True         , ( 0,  0,0) ),
]


id2color = {label.id: np.asarray(label.color) for label in labels}

def preprocess_frame(frame):
    frame = cv2.resize(frame, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    frame = frame.astype('float32') / 255.0
    return frame

def postprocess_image(image):
    pred = np.squeeze(np.argmax(image, axis=-1))
    output_image = np.zeros([IMAGE_SIZE[0], IMAGE_SIZE[1], 3])

    for row in range(IMAGE_SIZE[0]):
        for col in range(IMAGE_SIZE[1]):
            output_image[row, col, :] = id2color[pred[row, col]]

    output_image = output_image.astype('uint8')
    return output_image

def draw_rectangle_video(frame, x, y, width, height):
    if has_black_color(frame, x, y, width, height):
        rectangle_color = (0, 0, 255)
    else:
        rectangle_color = (0, 255, 0)

    cv2.rectangle(frame, (x, y), (x + width, y + height), rectangle_color, 2)

    return frame

def has_black_color(frame, x, y, width, height):
    for i in range(y, y + height):
        for j in range(x, x + width):
            pixel = frame[i, j]
            if pixel[0] == 0 and pixel[1] == 0 and pixel[2] == 0:
                return True
    return False

video = cv2.VideoCapture(0)

#prev_time = time.time()

while video.isOpened():
    ret, frame = video.read()

    if not ret:
        break

    #curr_time = time.time()
    #time_diff = curr_time - prev_time

    # Capture one frame per second
    #if time_diff >= 1.0:
        #prev_time = curr_time

    preprocessed_frame = preprocess_frame(frame)
    input_image = np.expand_dims(preprocessed_frame, axis=0)
    prediction = model.predict(input_image)
    processed_image = postprocess_image(prediction)

    print(processed_image.shape)

    output_frame = draw_rectangle_video(processed_image, 56, 56, 30 , 30)

    output_frame = cv2.resize(output_frame, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    cv2.imshow('Webcam and Output', output_frame)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()