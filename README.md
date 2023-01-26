In this task, a neural network was trained to solve the problem of determining the degree of fullness of a city trash tank from a image.

The image-based model recognized 3 classes.

1. Filled tank (it is visually obvious that the tank is full)
2. Half-empty tank (visually it can be seen that several bags from a standard household trash can can be put into the tank)
3. Empty tank (no trash is visible in the tank, while at least 25% of the internal space of the tank is visible)

This model can be used to optimize the work of city services. For example, the fill level of a tank can be determined based on street camera images that capture city tanks at a suitable angle. Relevant information can be automatically aggregated along with the location of the tank and taken into account when planning the routes of garbage trucks, or cause an unscheduled call / cleaning of the tank.
As a result, the garbage can be removed more quickly, if necessary, without unnecessary garbage trucks trips.

For the purposes of training the neural network, a manually assembled dataset was used. City tanks with varying degrees of fullness were filmed on a mobile phone, and then sorted into classes. The videos were cut into separate frames, and additionally filtered (thinned out) so that strongly similar frames were not used during training.

Only standard cubic tanks were used.

The videos were shot taking into account the fact that the tank should be in the center of the frame and take up most of it (> 50%). For the training task, frames that did not meet the criteria were removed, obviously broken, closed, non-standard tanks were not shot.

Total number of images: 2193 photos (763 empty, 1270 full, 160 half-empty).

For the purposes of training, validation, testing, the dataset was divided into training, validation, and test sets.
1. TRAIN SIZE: 1775
2. VALID SIZE: 220
3. TEST SIZE: 198

Since the classes were not balanced, StratifiedShuffleSplit from sklearn was used for splitting.

The model was trained on tensorflow 2.10.1, on one local GPU, within a notebook: model_train.ipynb.

Final training script: train_script.py.

In order to recreate the environment, need to install conda: https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html

Then use command to create an environment from the conda_environment.yml file: conda env create -f conda_environment.yml

To activate the environment, use the command: conda activate env_name

To containerize the application, docker-compose was used based on files:
1. image-gateway.dockerfile
2. image-model.dockerfile
3. Pipfile.lock
4. Pipfile
5. gateway.py
6. proto.py
7. docker-compose.yaml

Commands for preparation:
1. docker build -t capstone-2-gateway:002 -f image-gateway.dockerfile .
2. docker build -t capstone-2-model:xception-v4-001 -f image-model.dockerfile .

Run command: docker-compose up

The running application can be tested using the test.py script: python test.py