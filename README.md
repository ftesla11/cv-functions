# How to run

All the implemented functions are found in utilities.py

To run the code for displaying the appropriate outputs corresponding to the assignmnent questions, use the jupyter notebook main.ipynb. Run the cells in order.

The first cell imports the functions from utilities.py. From then on, any ICV_function() can simply be called in the notebook


# Dataset
The code uses the provided dataset (Dataset A, B, C). To run the code with the exact same outputs as the report, do not change any paths and variables and make sure the dataset folder named as 'Dataset' is located in the same directory as the python notebook. The images and videos are loaded using a hardcoded path, hence if the dataset folder does not have the same structure, or different datasets are desired to be used, change the path arguments in main.ipynb.


# Figures
The folder figures hold all the diagrams that are displayed when running the jupyter notebook. The subfolders inside figures are split into the topics according to the corresponding question, such as transformations and convolutions for question 1 and 2 respectively.