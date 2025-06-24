Weekend of 06/21/2025 project. Explores concepts related to self-driving cars in a simulated environment. Involves:
- Simulating the road, cars, and pedestrians
- Rendering 2d and 3d representations of the world using PyGame and OpenGL
- Extracting a dataset from the simulation, using it to train a convolutional neural network that can be used to control a car, then evaluating the qualitative performance of the network

An approximate system diagram is as follows:

![image](https://github.com/user-attachments/assets/48e26543-721c-4747-8fe3-0af79e6fd389)

Here's a screenshot of the 3d renderer (blue is car, red is pedestrian):
![image](https://github.com/user-attachments/assets/204be4fd-2f56-4447-a830-c12c4edb36bc)

Here's a screenshot of the 2d renderer:
![image](https://github.com/user-attachments/assets/48f158ab-2939-4d0b-b672-7794b8d870f3)



There are some example videos in this repository:
- The 2d renderer: "2d simulation rendering.mp4"
- The 3d renderer: "3d simulation rendering.mp4"
- Examples of the car controlled by the neural network:
- With a smaller neural network and few samples: "car performance 1 - failure.mp4" (car drives off the road pretty fast)
- After training on more samples and adding dropout: "car performance 2 - semi-successful.mp4" (car can drive on the road for a moderate period)
- After more training epochs: "car performance 3 - successful.mp4" (car drives on the road for a while)
