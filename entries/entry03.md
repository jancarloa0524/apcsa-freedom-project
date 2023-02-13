# Entry 3 - `import cv2`
### 2/13/23

In my last entry, I had spent a lot of time talking about learning Python, and now that I pretty much have, I am beginning to learn about [OpenCV](https://docs.opencv.org/4.7.0/d6/d00/tutorial_py_root.html) and machine learning. In order to start, I had to learn how to install OpenCV on my system and run it in a *virtual environmet*, which is where a fresh Python executable and libraries are installed that do not conflict with the libraries installed on my system, which keeps all my code clean.

#### **Basic Image/Video Reading and Processing**

I learned about quite a few of OpenCV's basic functions, some of which can be summed up in the following code snippet:
``` python
import cv2
# Read the image file
img = cv2.imread('image.jpg')
# Show "img" in a window called "Image"
cv2.imshow("Image", img)
# Wait for the user to press a key before executing more code
cv2.waitKey(0)
# Destroy all windows opened by OpenCV
cv2.destroyAllWindows()
```

There were many, many more, but so far, what I can gather is that there are many basic functions that can read both images and videos, as well as manipulate them. 

*Note: Reading videos is similiar, but since a video is a series of images, we have to display the video as a series of images with a while loop. Interestingly, we can also use this to open up a live recording of our webcam (0 means use internal webcam):* `capture = cv2.VideoCapture(0)`

#### **Manipulating Images**

Manipulating images is interesting, because in OpenCV, images are BGR (blue, green, red), rather than RGB. We can easily change certain properties of images with functions like `cv2.cvtColor(img, flag)`, where flag can be a value like `cv2.COLOR_BGR2GRAY`. This is what that code might look like:
``` python
newImg = cv2.cvtColor('img.jpg', cv2.COLOR_BGR2GRAY)
# We can also save this image onto the disk!
cv2.imwrite("/newFolder/newImg.jpg", newImg)
```

We can even draw on images! Doing this reminded me of learning p5.js in 11th grade.
``` python
img = cv2.imread('img.png')

cv2.rectangle(img,            # where to draw
              (200, 100),     # top-left
              (500, 400),	  # bottom-right
              (255, 0, 0),    # color
              2)              # thickness
```

#### Supervised Machine Learning

My introduction to machine learning was with *supervised learning*, which includes data that is already labelled, so it has definitive inputs expected by the algorithm. The algorithm learns from this data, and is capable of producing a function which maps the given inputs to their expected outputs. 

The specific algorithms I learned about were K-Nearest Neighbor, and Support Vector Machine Algorithms. 

**Support Vector Machines (SVM):** This algorithm finds a boundary which best divides a dataset into the number of classes which exist in it. If the data is difficult to seperate with a linear line, then the data will essentially look at more features to create a more linearly seperable graph. 

**K-Nearest Neighbor Algorithm (KNN):** Simply put, a 'similarity score' is computed by comparing a test image against other images in a data set, and places the test image in a category based on the similarity score. 

In order to learn about these algorithms, I used the *Quick, Draw!* dataset, which has thousands of hand made sketches of various objects, in a 28 by 28 gray scale flattened image format. "Flattened" means the images are not 2 dimensional, they are currently 1 dimensional arrays containing 784 elements. 

I had to first visualize the dataset to really understand what I was dealing with.

``` python
import cv2

import numpy as np # numpy is a library used for dealing with large arrays and images
# loads the dataset
dataset = np.load('data.npy')
# Print the length of the dataset
print("Num of elements:", len(dataset))
# Reshape the dataset by appending the reshaped images into a new list
reshapedDataset = []
for x in range(20):
    reshapedDataset.append(np.reshape(dataset[x], (28, 28)))
# Display these images on screen
for x in range(len(reshapedDataset)):
    cv2.imshow(str(x), reshapedDataset[x])

cv2.waitKey(0)
cv2.destroyAllWindows()
```

Finally, I had to actually test an algorithm. For this one, I decided to test KNN, so I only needed to import KNN. To keep this simple, the goal was to see the accuracy score of this algorithm after training it. 80% of the dataset has been chosen for testing, and 20% for testing. The steps were as follows:

1. Load datasets and take samples.
2. Concatenate the datasets
3. Normalize the datasets with interpolation, a method of estimation based on given data
4. Prepare the data for the training/testing split
5. Train the KNN algorithm
6. Test the KNN algorithm's accuracy with the test portion of the dataset.

This is what that code looked like:
``` python
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import accuracy_score

N_SAMPLES = 1000
TEST_SIZE = 0.2
APPLE = 0
BANANA = 1

def normalize(data):
    return np.interp(data, [0, 255], [-1, 1])

apples_full = np.load('dataset/apple.npy')
bananas_full = np.load('dataset/banana.npy')

apples = apples_full[:N_SAMPLES]
bananas = bananas_full[:N_SAMPLES]

dataset = np.concatenate((apples, bananas))

dataset = normalize(dataset)

labels = [APPLE] * N_SAMPLES + [BANANA] * N_SAMPLES

x_train, x_test, y_train, y_test = tts(dataset, labels, test_size=TEST_SIZE)

clf = KNN()

clf.fit(x_train, y_train)

preds = clf.predict(x_test)

accuracy = accuracy_score(y_test, preds)
print("Accuracy:", accuracy)
```

### EDP

At this point, I feel I am transitioning into *Stage 3: Brainstorming Possible Solutions*, since I am finally learning about Machine Learning Algorithms and can begin to have an idea of what exactly I want to do. Since videos are just a sequence of images, I could use SVM or KNN algorithms to categorize videos of different kicks, however I'm sure this would take a very, very long time to train. 
### Skills

I feel I have really worked on my *Growth Mindset*, because this is taking a long time and a lot of effor to learn about, but I have to say that I had a lot of fun and it has felt rewarding teaching myself Python and a bit of machine learning.

I also feel I have worked on *Attention to Detail*, because I have been taking very organized notes that pay attention to pretty much every function and new syntax that I learn about. I pay close attention to the way the syntax should be written, and make sure I adhere to it.
### Conclusion

Overall, learning about Machine Learning with Python and OpenCV has been a very fun process, it isn't too difficult to grasp so far, but there has been a lot to grasp. I look forward to learning more, and hopefully learning at a faster pace.

[Previous](entry02.md) | [Next](entry04.md)

[Home](../README.md)