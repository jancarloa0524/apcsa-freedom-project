# Entry 6 - `model.save(proto_2.h5)`
### 6/4/23

After completing my **minimum viable product (MVP)** in my last entry, I wasn't exactly satisfied with the way detections were actually being made. While my program did make detections, it wasn't necessarily accurate, since it only checked a single frame to see if a move had been made. The problem with this is that movement doesn't happen over a single frame. So, this time I learned just a bit about something called [TensorFlow](), and learned about detecting complete motion through this [tutorial](), in order to create my [2nd Prototype]()!

Later in this blog, I will talk about my experience giving my in-class [presentation](https://docs.google.com/presentation/d/1FeGKmW4rUroUsBJIB_JAE8QlwqM7RTan5QHmJDLA9gs/edit?usp=sharing), and also my experience giving my elevator pitch during the [SEP Freedom Project Expo](https://docs.google.com/document/d/1KWElCYnIoxcg4EEcW3FhXSfx1K0eptsTUh54YwEj-o4/edit?usp=sharing).

## Beyond the MVP

While the previous prototype used [Sci-Kit Learn]() in order to train a model, which only uses classicial machine learning algorithms, this prototype used a more complicated **Neural Network Deep Learning algorithm**. This is done with [TensorFlow](), a free and open source machine learning and AI library developed by Google. Nueral Networks are layered networks of neurons that learn and process data in order to create a more accurate output each time it is trained. 

### How Does This Work?

To keep it simple, I will briefly explain and show some snippets of code. 

First off, I now needed to collect data **every 30 frames**, labelling it an action such as a jab or cross. In the final program, it will check the past 30 frames, up to the current frame, for the given action. 

``` python
# path for exported numpy arrays
DATA_PATH = os.path.join('Data')
# Actions we detect
actions = np.array([...])
# thirty videos worth of data
num_sequences = 30
# 30 frame length videos
sequence_length = 30
# storing each of our 30 frames (as numpy arrays) in different folders
for action in actions:
	for sequence in range(num_sequences):
		try:
			os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
			# create a folder called Data, then a subfolder for the action, then a subfolder for the sequence
		except:
			pass
...
# THE FOLLOWING COMES LATER WHEN WE ARE LOOPING THROUGH 30 FRAMES OF DATA FOR EACH SEQUENCE
# save each frame as a np array, resulting in 30 np arrays for each sequence
pose_keypoints = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num)) # where we are saving our frame
np.save(npy_path, pose_keypoints) # save the frame
```

Later on, we use TensorFlow to actually train the data, which looks a bit like this:

``` python
# this is the nueral networks layers; LSTM layers are best for images, Dense layers then condense the data
model = Sequential([
    LSTM(64, return_sequences = True, activation='relu', input_shape=(30,132)),
    LSTM(128, return_sequences = True, activation='relu'),
    LSTM(64, return_sequences = False, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(actions.shape[0], activation='softmax')
    ])
...
# Later on, we "fit" our data in the model to create a model that processes real time data
model.fit(X_train, y_train, epochs=70, callbacks=[tensorboard_callback]) # train our model
```

This is just a bit of what the new code now looks like, and it was a very complicated thing to learn and understand. TensorFlow was what really made it difficult, because it determined why the data is formatted the way it is, and how the program will run. Overall, programming this was not so difficult since I did follow a tutorial that helped me to write and understand TensorFlow, however if I want to further modify and improve the code, I feel I'll need to learn a lot more about the tools I have at my disposal, that I did not have the time to learn about before. 

## In-Class Presentations:

Overall, I'd say the in-class presentations went very well. I really enjoyed putting together my presentation, becuase it walked the audience through my journey and the things I learned and the obstacles I crossed.

I also remembered that in SEP 11th Grade, my presentation was very long, and while enjoyable, I did not want to have a long, drawn out presentation again, so I structured my presentation around presenting complex ideas in a more simple manner, while only showing off some code that I could explain quickly. 

## Expo Elevator Pitch

The expo elevator pitch was an exciting and, at times, overwhelming experience. I was very excited to present my project to others, so I went all out with a duel moniter set-up, and having enough space to actually demonstrate my project and allow others to try it. I even brought a kicking target!

Writing my elevator pitch was easy becuase of the fact that I was so passionate about my project, and I relied on the demonstration to further garner interest in my project, as I explained it. On the other screen, I had videos displaying the working model, just in case. 

At times, I noticed lots of people watching demonstrations of my project and listening to me talk about it. I remember looking up to see so many people that I felt I wasn't even sure what to do, but I just kept on talking!

## EDP

Currently in the **Engineering Design Process**, I am at stage 7, *Improve as needed*, and stage 8, *Communicate the results*. I am passionate about improving this project and giving it better functionality. I am also a Freedom Project Finalist, meaning I'll have to present the project to the rest of the Software Engineering Program. 

## Skills

This time, I really improved my **Attention to detail**. While creating my project, I looked for ways to make training my model easier, so I imported a library that allows me to play sounds as it's testing to let me know when to perform an action and when to change the type of action im performing. I also paid a lot of attention to the detail in my presentations and elevator pitch to provide the highest quality presentation I can. I alos improved my **Creativity**, since I really focused on how I can present my project in a way that is appealing, fun, and interactive for the audience.

## Conclusions

I had a ton of fun working on this project. It was a lot of learning, and there is still a lot more learning to go, but I have grown to understand something that I couldn't even explain a year ago. I want to continue to work on my project and improve on it as needed, because I want to provide a tool that can truly help martial artists improve their skills. 

[Previous](entry05.md)

[Home](../README.md)