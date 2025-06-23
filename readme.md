This is a project that explores concepts related to self-driving cars in a simulated environment. It involves:
- Simulating the road, cars, and pedestrians
- Rendering 2d and 3d representations of the world using PyGame and OpenGL
- Extracting a dataset from the simulation and using it to train a convolutional neural network that can be used to control a car, then evaluating the performance of this network

An approximate system diagram is as follows:
![image](simple system sketch.png)

Here are some example videos:
The 2d renderer:
<video src="2d simulation rendering.mp4" width="320" height="240" controls></video>
The 3d renderer:
<video src="3d simulation rendering.mp4" width="320" height="240" controls></video>
Examples of the car controlled by the neural network:
With a smaller neural network and few samples:
<video src="car performance 1 - failure.mp4" width="320" height="240" controls></video>
With a smaller neural network and more samples:
<video src="car performance 2 - semi-successful.mp4" width="320" height="240" controls></video>
With a slightly larger neural network, more samples, and some randomness introduced into the simulation:
<video src="car performance 3 - semi-successful.mp4" width="320" height="240" controls></video>