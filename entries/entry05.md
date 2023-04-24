# Entry 5 - `Probability = 0.5`, `Position = Completed`
### 4/17/23

In my last entry, I was still learning about machine learning and how it works. While much of it was helpful in understanding how my project would end up turning out, I needed to speed up my progress and move ahead to creating my MVP. I quickly learned as much as I could about [MediaPipe](https://mediapipe.dev/), [Holistic Detection](https://github.com/google/mediapipe/blob/master/docs/solutions/holistic.md) and [Pose Detection](https://github.com/google/mediapipe/blob/master/docs/solutions/pose.md) through the following tutorials:
- [Hollistic Model (Helped me with creating pose detection)](https://www.youtube.com/watch?v=pG4sUNDOZFg)
- [AI Body Language Decoder (Helped me with learning how to train a model)](https://www.youtube.com/watch?v=We1uB79Ci-w)

All the code for the MVP I am about to explain can be found in this [repo](https://github.com/jancarloa0524/tkd-pose-detection/tree/main).

## Finishing this MVP

To start off, I used the **Mediapipe Hollistic Detection** tutorial to create the **pose detection** model. I did this because Hollistic Detection includes full face detection, complete hand detection, and complete body pose detection. However, it calls all three detections, when I can just use the individual ones. This is important because of the fact even if I don't *render* the face and hand detection with the Holistic Detection model, it still *detects* the key points, meaning less performance. I need to only capture body pose, which has which only have the following key points:

![Pose Landmarks](../img/pose_landmarks.png)

- Hand pose captures quite a few key points as well, since hands have complicated joints, but face detection is a grid of many, many keypoints. 

In order to visualize this model on a person, I had to use OpenCV's webcam rendering capabilities, and import Mediapipe's pose detection model and rendering solutions.

``` python
import mediapipe as mp
import cv2

mp_drawing = mp.solutions.drawing_utils # Rendering Solutions
mp_pose = mp.solutions.pose # Holisitic Model Solutions

cap = cv2.VideoCapture(0) # OpenCV Webcam Capture (0 = Webcam feed)
```

I then had to initiate the pose model, which inlcudes a while loop to render every frame of the webcam feed. The user will press 'q' to end the feed, and the program will shut down. 

``` python
# Initiate holistic model
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened(): # while webcam is running
        ...
        k = cv2.waitKey(10)
        if k == ord('q'):
            break # break out of feed upon key-press

cap.release() 
cv2.destroyAllWindows() # shut down program
```

Once this is done, we can now start including pose detections within our while loop:

``` python
while cap.isOpened(): # while webcam is running
        ret, frame = cap.read()

        # Recolor
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Holistic Detections
        results = pose.process(img)

        # Color back
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Render Pose Detections
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius = 4), mp_drawing.DrawingSpec(color=(0,0,0), thickness = 2, circle_radius = 2))

        cv2.imshow("Feed", img)
        
        k = cv2.waitKey(10)

        if k == ord('q'):
            break
```
### Training a Model

Once we have a working pose detection model, we then have to create a spreadsheet that holds the data for our landmarks.

``` python
# Export landmarks to CSV
import csv # for working with CSV file
import os # for working with files
import numpy as np # works with array mathematics

...

# Capture landmarks to CSV file
num_coords = len(results.pose_landmarks.landmark)

landmarks = ['class']
for val in range(1, num_coords + 1):
    landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)] # set up format of CSV file

with open('coords.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(landmarks) # write first row of landmarks
```

We then also have to have a way to actually capture and write our data to the CSV file, which is done while running the pose detection model.

``` python
try:
    class_name = "class_name_here"
    # Extract pose landmarks
    pose_results = results.pose_landmarks.landmark
    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose_results]).flatten())
    # Append class name for exporting
    pose_row.insert(0, class_name)
    # Export to CSV
    with open('coords.csv', mode='a', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow(pose_row)
except:
    pass
```

Within `class_name`, we hold the name of our class everytime we test a new position, classifying our data. 

Once we have all our data collected, we must train an algorithm based off of it. This code can be seen in `trainer.py`, but to explain it briefly, we first take our data and create training/testing splits. We then decide on an algorithm and train said algorithm, outputting it as a file that can be read by `model.py`, which is where all the sample code is coming from. 

Finally, once we have our working algorithm, we can run it and use it to make detections:

``` python
# Making Detections
import pandas as pd # working with tabular data
import pickle # library for saving and opening models on disks

# open model for making detections
with open('tkd.pkl', 'rb') as f:
                model = pickle.load(f)
...
# Making Detections
X = pd.DataFrame([pose_row]) 
tkd_class = model.predict(X)[0] # predict and extract first X values
tkd_class_prob = model.predict_proba(X)[0]
print(tkd_class, tkd_class_prob)

# Get status box
cv2.rectangle(img, (0,0), (250, 60), (0,0,0), -1)
# Display Class
cv2.putText(img, 'POSITION'
            , (100, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
cv2.putText(img, tkd_class.split(' ')[0]
            , (110, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

# Display Probability
cv2.putText(img, 'PROB'
            , (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
cv2.putText(img, str(round(tkd_class_prob[np.argmax(tkd_class_prob)],2))
            , (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
```

Much of the code seen, such as anything that starts with `cv2`, has already been something I learned. It was very interesting to see past concepts and tools come full circle later on. 

### EDP

I am currently in **Stage 5/6: Creating/Testing and evaluating prototype**. I'd say that I am here simply because the prototype is working, however it is the bare minimum. My completed idea involved more detections that can be tested by anyone, so I will be writing and rewriting code, and testing and evaluating the prototype over and over. The reason why I wouldn't say I'm at **Stage 7: Improving as needed**, is simply because I do not have a completed algorithm with all the kicks/punches I want it to recognize yet. It simply accomplishes the minimum of recognizing a jab and a snap-kick. 


### Skills

I'd say I rapidly improved my **How to Learn** skill, since I had to learn about Mediapipe and a few other libraries fairly quickl for this project. While I have not learned everything I need to know just yet, I have learnt the minimum in a fairly short period of time. 

I also worked on **Logical Reasoning**, figuring out that I could replace the Hollistic Detection with Pose Detection, which made my program run much faster and smoother, which can then also collect better data. 

### Conclusions

Overall, I'm very proud of what I have been able to accomplish over the past few months. It may seem like I had a lot of time, but in reality, I did not. I spent much of my spare time learning about Python and machine learning, believing that I might not make it in time to creating an MVP. However, hard work and determination paid off, especially all the research I had done. It made learnning about Hollistic and Pose detection very easy, and I look forward to further improving my product. 

[Previous](entry04.md) | [Next](entry06.md)

[Home](../README.md)