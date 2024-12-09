import os
import shutil
import re
import dlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier as ORC
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA

# Disable warnings
def warn(*args, **kwargs):
    pass

import warnings
warnings.warn = warn

# ---------------------------
# Organize Images into Folders
# ---------------------------

# Path to the "faces" folder on the desktop
faces_folder = os.path.expanduser("/Users/danabenish/Desktop/mobilebio project/FaceDisguiseDatabase/FaceAll_cropped")

# List all .jpg files in the faces folder and sort them
images = [f for f in os.listdir(faces_folder) if f.endswith(".jpg")]
images.sort()

# Regular expression to extract the number from the filename
pattern = re.compile(r"(\d+)")

# List to keep track of images by their numerical order
image_numbers = []
for img in images:
    match = pattern.search(img)
    if match:
        number = int(match.group(1))
        image_numbers.append((number, img))

# Sort the images by their extracted number (in case they weren't sorted already)
image_numbers.sort(key=lambda x: x[0])

# Variables to keep track of folder numbering
folder_num = 1

# Start processing the images
index = 0
while index < len(image_numbers):
    # Start with the current image number
    current_image = image_numbers[index]
    current_number = current_image[0]

    # Create the folder for the current group
    folder_name = os.path.join(faces_folder, f"face{folder_num}")
    os.makedirs(folder_name, exist_ok=True)

    # Add images to the folder as long as their numbers are within the current + 5 range
    group = [current_image]
    index += 1

    # Check if the next images are within current_number + 5
    while index < len(image_numbers) and image_numbers[index][0] <= current_number + 5:
        group.append(image_numbers[index])
        index += 1

    # Move all images in the current group to the folder
    for _, img in group:
        shutil.move(os.path.join(faces_folder, img), os.path.join(folder_name, img))

    # Increment the folder number for the next group
    folder_num += 1

print("Images have been organized into folders.")

# ---------------------------
# Extract Facial Landmarks
# ---------------------------

def distances(points):
    return [np.linalg.norm(p1 - p2) for p1 in points for p2 in points]

def get_bounding_box(rect):
    x, y, w, h = rect.left(), rect.top(), rect.right() - rect.left(), rect.bottom() - rect.top()
    return x, y, w, h

def shape_to_np(shape, num_coords, dtype="int"):
    coords = np.array([(shape.part(i).x, shape.part(i).y) for i in range(num_coords)], dtype=dtype)
    return coords

def get_landmarks(images, labels, save_directory="landmarks_folder", num_coords=68, to_save=True):
    print("Getting %d facial landmarks" % num_coords)
    landmarks = []
    new_labels = []
    img_ct = 0
    predictor_path = '/Users/danabenish/Desktop/mobilebio project/Loading Images and Extracting Landmarks 2/shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)

    for img, label in zip(images, labels):
        img_ct += 1
        print(f"Processing image {img_ct}/{len(images)}: {label}")
        detected_faces = detector(img, 1)

        if not detected_faces:  # No faces detected
            print(f"No faces detected for image {label}. Using dummy distances.")
            new_labels.append(label)  # Still append the label
            landmarks.append(dist)  # Append previous distances
        else:
            for d in detected_faces:
                new_labels.append(label)
                x, y, w, h = get_bounding_box(d)
                points = shape_to_np(predictor(img, d), num_coords)
                
                # Ensure the points are returned as (num_coords, 2) shape
                if points.shape != (num_coords, 2):
                    raise ValueError("Landmark shape mismatch: Expected (%d, 2) but got %s" % (num_coords, points.shape))
                
                dist = distances(points)
                landmarks.append(dist)

        if to_save and detected_faces:
            # Save images with the same label, but append a unique index to prevent overwriting
            img_filename = str(label) + f"_{img_ct}.png"
            for (x_, y_) in points:
                cv2.circle(img, (x_, y_), 1, (0, 255, 0), -1)
            plt.figure()
            plt.imshow(img)
            os.makedirs(save_directory, exist_ok=True)
            plt.savefig(os.path.join(save_directory, img_filename))
            plt.close()

        if img_ct % 50 == 0:
            print("%d images with facial landmarks completed." % img_ct)

    # Convert landmarks and labels into numpy arrays
    print("Total images processed:", len(images))
    print("Total landmarks extracted:", len(landmarks))
    print("Total labels:", len(new_labels))

    return np.array(landmarks), np.array(new_labels)

# ---------------------------
# Load Ground Truth Data from .txt Files
# ---------------------------
# Path to the "Ground_Truth" folder
ground_truth_folder = '/Users/danabenish/Desktop/mobilebio project/FaceDisguiseDatabase/Ground_Truth'

# List all .txt files in the Ground_Truth folder
txt_files = [f for f in os.listdir(ground_truth_folder) if f.endswith('.txt')]

# Create an empty list to store the ground truth data
ground_truth_data = []

# Loop through each .txt file and parse its content
for txt_file in txt_files:

    try:
        # Read the content of the .txt file
        with open(os.path.join(ground_truth_folder, txt_file), 'r') as file:
            line = file.readline().strip()
            
            # Split the line by commas
            parts = line.split(',')
            
            # Handle special cases and convert invalid values to 1
            for j in range(3, len(parts)): 
                try:
                    value = int(parts[j])
                    # Ensure the value is either 0, 1, or 2, otherwise set it to 1
                    if value not in [0, 1, 2]:
                        parts[j] = '1' 
                except ValueError:
                    # If it's not an integer, set it to 1
                    parts[j] = '1'

            # Extract the relevant ground truth data
            filename = txt_file.replace('.txt', '.jpg')
            width = int(parts[1])
            height = int(parts[2])
            sex = int(parts[3])
            skin_color = int(parts[4])
            mustache = int(parts[5])
            beard = int(parts[6])
            glasses = int(parts[7])
            hat = int(parts[8])
            
            # Append the parsed data as a tuple
            ground_truth_data.append((filename, width, height, sex, skin_color, mustache, beard, glasses, hat))
    
    except ValueError as e:
        print(f"Error processing file {txt_file}: {e}")
        continue  # Skip the problematic file and continue with the next

# Convert the list of tuples to a pandas DataFrame
ground_truth_df = pd.DataFrame(ground_truth_data, columns=['filename', 'width', 'height', 'sex', 'skin_color', 'mustache', 'beard', 'glasses', 'hat'])

# Print the DataFrame to check if the data is loaded correctly
print(ground_truth_df.head())


# ---------------------------
# Assign hair Labels from Ground Truth
# ---------------------------

# Define hair mapping
hair_mapping = {0: 0, 1: 1, 2: 2} 

# Initialize list for image files
image_files = []
labels = []

for root, dirs, files in os.walk(faces_folder):
    for file in files:
        if file.endswith(".jpg"): 
            image_path = os.path.join(root, file) 
            folder_name = os.path.basename(root)  

            # Append the image path and folder name (as the label) to the respective lists
            image_files.append(image_path)
            labels.append(folder_name) 

# Now `image_files` contains the image paths and `labels` contains the hair labels

# Initialize hair labels list
facial_hair_values = []

# Assign hair labels based on ground truth
for img_file in image_files:
    filename = os.path.basename(img_file)
    # Get the corresponding row in the ground truth dataframe
    matching_row = ground_truth_df[ground_truth_df['filename'] == filename]
    
    if not matching_row.empty:
        mustache = matching_row['mustache'].values[0]
        beard = matching_row['beard'].values[0]

        
        # Ensure that hair value is either 0 or 1; otherwise set to 1
        if mustache == 0 and beard == 0:
            facial_hair_values.append(0)  
        elif mustache == 1 or beard == 1:
            facial_hair_values.append(1) 
        else:
            # If hair_value is anything other than 0 or 1, set to 1 (hair)
            print(f"Unexpected hair value for {filename}: {facial_hair_values}. Setting to 1.")
            facial_hair_values.append(1) 
    else:
        print(f"No ground truth match for {filename}")
        facial_hair_values.append(0)

# Convert to numpy array
facial_hair_values = np.array(facial_hair_values)


# Print unique values and counts for hair labels
print(np.unique(facial_hair_values, return_counts=True))

# ---------------------------
# Load Images and Extract Landmarks
# ---------------------------

# Load images

images = [cv2.imread(img_file) for img_file in image_files]
print(len(images))
print(len(labels))

#x, y = get_landmarks(images, labels, num_coords=68)

# ---------------------------
# Filter Images Based on hair Labels
# ---------------------------
x = np.load("X-68-Caltech.npy") 
y = np.load("y-68-Caltech.npy")

#hair_labels = np.load("hair-labels")
mask_no_hair = facial_hair_values == 0
X_no_hair = x[mask_no_hair]
y_no_hair = y[mask_no_hair]

X_hair = x[~mask_no_hair]
y_hair = y[~mask_no_hair]

np.save("X-68-Caltech.npy", x)
np.save("y-68-Caltech.npy", y)
#np.save("X-5-Caltech.npy", x)
#np.save("y-5-Caltech.npy", y)
#np.save("hair-labels.npy", hair_labels)


print("Landmark files saved.")
print("Shape of x:",x.shape)
print("Shape of X_no_hair:", X_no_hair.shape)
print("Shape of y_no_hair:", y_no_hair.shape)
print("Shape of X_hair:", X_hair.shape)
print("Shape of y_hair:", y_hair.shape)



# Initialize classifiers
knn = KNeighborsClassifier(n_neighbors=10, weights='distance', algorithm='auto')
svc = SVC(C=1, class_weight='balanced', probability=True, kernel = 'linear')  # Enable probability estimation

# Wrap the classifiers with OneVsRestClassifier
clf = (knn)
clf2 = (svc)

# ---------------------------
# Train on no-hair, test on hair
# ---------------------------


# Initialize metrics storage
gen_scores = []
imp_scores = []

clf.fit(x, y)
print("done fitting 1")
clf2.fit(x, y)
print("done fitting 2")

# Test classifiers
knn_scores = clf.predict_proba(X_hair)
svc_scores = clf2.predict_proba(X_hair)
print("predicted probs")

# Combine scores (average)
combined_scores = (knn_scores + svc_scores) / 2
classes = clf.classes_
matching_scores = pd.DataFrame(combined_scores, columns=classes)
print("matched scores")

knn_test_predictions = clf.predict(X_hair)
svc_test_predictions = clf2.predict(X_hair)
knn_test_accuracy = accuracy_score(y_hair, knn_test_predictions)
svc_test_accuracy = accuracy_score(y_hair, svc_test_predictions)
print(f"KNN Test Accuracy: {knn_test_accuracy:.2f}")
print(f"SVC Test Accuracy: {svc_test_accuracy:.2f}")

for i in range(len(y_hair)):  
    try:  
        scores = matching_scores.loc[i]
        mask = scores.index.isin([y_hair[i]])
        gen_scores.extend(scores[mask])
        imp_scores.extend(scores[~mask])
    except:
        print("not found")


print("Length of scores:")
print(len(gen_scores))
print(len(imp_scores))

class Evaluator:
    """
    A class for evaluating a biometric system's performance.
    """

    def __init__(self, 
                 num_thresholds, 
                 genuine_scores, 
                 impostor_scores, 
                 plot_title, 
                 epsilon=1e-12):
        """
        Initialize the Evaluator object.

        Parameters:
        - num_thresholds (int): Number of thresholds to evaluate.
        - genuine_scores (array-like): Genuine scores for evaluation.
        - impostor_scores (array-like): Impostor scores for evaluation.
        - plot_title (str): Title for the evaluation plots.
        - epsilon (float): A small value to prevent division by zero.
        """
        self.num_thresholds = num_thresholds
        self.thresholds = np.linspace(-0.1, 1.1, num_thresholds)
        self.genuine_scores = genuine_scores
        self.impostor_scores = impostor_scores
        self.plot_title = plot_title
        self.epsilon = epsilon

    def get_dprime(self):
        """
        Calculate the d' (d-prime) metric.

        Returns:
        - float: The calculated d' value.
        """
        '''
        D Prime via Lecture 5 Slides
        '''
        x = np.sqrt(2) * np.abs(np.mean(self.genuine_scores) - np.mean(self.impostor_scores))
        y = np.sqrt(np.std(self.genuine_scores)**2 + np.std(self.impostor_scores)**2)
        
        return x / (y + self.epsilon)

    def plot_score_distribution(self):
        """
        Plot the distribution of genuine and impostor scores.
        """
        plt.figure()
        
        # Plot the histogram for genuine scores
        plt.hist(
            # Provide genuine scores data here
            # color: Set the color for genuine scores
            # lw: Set the line width for the histogram
            # histtype: Choose 'step' for a step histogram
            # hatch: Choose a pattern for filling the histogram bars
            # label: Provide a label for genuine scores in the legend
            self.genuine_scores,
            color = "green",
            lw = 1,
            histtype = 'step',
            hatch = '/',
            label = "Genuine Scores"
            
        )
        
        # Plot the histogram for impostor scores
        plt.hist(
            # Provide impostor scores data here
            # color: Set the color for impostor scores
            # lw: Set the line width for the histogram
            # histtype: Choose 'step' for a step histogram
            # hatch: Choose a pattern for filling the histogram bars
            # label: Provide a label for impostor scores in the legend
            self.impostor_scores,
            color = "red",
            lw = 1,
            histtype = 'step',
            hatch = "|",
            label = "Impostor Scores"
        )
        
        # Set the x-axis limit to ensure the histogram fits within the correct range
        plt.xlim([-0.05, 1.05])
        
        # Add grid lines for better readability
        plt.grid(color='gray', linestyle='--', linewidth=0.5)
        
        # Add legend to the upper left corner with a specified font size
        plt.legend(
            # loc: Specify the location for the legend (e.g., 'upper left')
            # fontsize: Set the font size for the legend
            loc = 'upper left',
            fontsize = '10'
        )
        
        # Set x and y-axis labels with specified font size and weight
        plt.xlabel(
            # Provide the x-axis label
            # fontsize: Set the font size for the x-axis label
            # weight: Set the font weight for the x-axis label
            'Score',
            fontsize = 12,
            weight ='bold'
        )
        
        plt.ylabel(
            # Provide the y-axis label
            # fontsize: Set the font size for the y-axis label
            # weight: Set the font weight for the y-axis label
            'Frequency',
            fontsize = 12,
            weight ='bold'
        )
        
        # Remove the top and right spines for a cleaner appearance
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        # Set font size for x and y-axis ticks
        plt.xticks(
            # fontsize: Set the font size for x-axis ticks
            fontsize = 10
        )
        
        plt.yticks(
            # fontsize: Set the font size for y-axis ticks
            fontsize = 10
        )
        
        # Add a title to the plot with d-prime value and system title
        plt.title('Score Distribution Plot\nd-prime= %.2f\nSystem %s' % 
                  (self.get_dprime(), 
                   self.plot_title),
                  fontsize=15,
                  weight='bold')
        
       
        # Save the figure before displaying it
        plt.savefig('score_distribution_plot_(%s).png' % self.plot_title, dpi=300, bbox_inches="tight")
        
        # Display the plot after saving
        plt.show()
        
        # Close the figure to free up resources
        plt.close()

        return

    def get_EER(self, FPR, FNR):
        """
        Calculate the Equal Error Rate (EER).
    
        Parameters:
        - FPR (list or array-like): False Positive Rate values.
        - FNR (list or array-like): False Negative Rate values.
    
        Returns:
        - float: Equal Error Rate (EER).
        """
        
        min_diff = float('inf')
        EER = 0
        
        
        # Add code here to compute the EER
        
        for i in range(len(FPR)):
                fpr_val = FPR[i]
                fnr_val = FNR[i]
        
                diff = abs(fpr_val - fnr_val)
        
                if diff < min_diff:
                    min_diff = diff
                    EER = (fpr_val + fnr_val) / 2  

        return EER

    def plot_det_curve(self, FPR, FNR):
        """
        Plot the Detection Error Tradeoff (DET) curve.
        Parameters:
         - FPR (list or array-like): False Positive Rate values.
         - FNR (list or array-like): False Negative Rate values.
        """
        
        # Calculate the Equal Error Rate (EER) using the get_EER method
        EER = self.get_EER(FPR, FNR)
        
        # Create a new figure for plotting
        plt.figure()
        
        # Plot the Detection Error Tradeoff Curve
        plt.plot(
            # FPR values on the x-axis
            # FNR values on the y-axis
            # lw: Set the line width for the curve
            # color: Set the color for the curve
            FPR,
            FNR,
            lw = 2,
            color = 'blue',
            label = "DET Curve"
        )
        
        # Add a text annotation for the EER point on the curve
        # Plot the diagonal line representing random classification
        # Scatter plot to highlight the EER point on the curve

        plt.text(EER + 0.07, EER + 0.07, "EER", style='italic', fontsize=12,
                 bbox={'facecolor': 'grey', 'alpha': 0.5, 'pad': 10})
        plt.plot([0, 1], [0, 1], '--', lw=0.5, color='black')
        plt.scatter([EER], [EER], c="black", s=100)
        
        # Set the x and y-axis limits to ensure the plot fits within the range 
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        
        # Add grid lines for better readability
        plt.grid(
            # color: Set the color of grid lines
            # linestyle: Choose the line style for grid lines
            # linewidth: Set the width of grid lines
            color = 'black',
            linestyle = '--',
            linewidth = '1'
        )
        
        # Remove the top and right spines for a cleaner appearance
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        # Set x and y-axis labels with specified font size and weight
        plt.xlabel(
            # 'False Pos. Rate': Set the x-axis label
            # fontsize: Set the font size for the x-axis label
            # weight: Set the font weight for the x-axis label
            'False Pos. Rate',
            fontsize = 12,
            weight = 'bold'
        )
        
        plt.ylabel(
            # 'False Neg. Rate': Set the y-axis label
            # fontsize: Set the font size for the y-axis label
            # weight: Set the font weight for the y-axis label
            'False Neg. Rate',
            fontsize = 12,
            weight = 'bold'
        )
        
        # Add a title to the plot with EER value and system title
        plt.title(
            # 'Detection Error Tradeoff Curve \nEER = %.5f\nSystem %s': Set the title
            # EER: Provide the calculated EER value
            # self.plot_title: Provide the system title
            # fontsize: Set the font size for the title
            # weight: Set the font weight for the title
            f'Detection Error Tradeoff Curve \nEER = {EER:.5f}\nSystem {self.plot_title}',
            fontsize = 16,
            weight = 'bold'
        )
        
        # Set font size for x and y-axis ticks
        plt.xticks(
            # fontsize: Set the font size for x-axis ticks
            fontsize = 12
        )
        
        plt.yticks(
            # fontsize: Set the font size for y-axis ticks
            fontsize = 12
        )
        
        # Save the plot as an image file
        plt.savefig(
            'det.png',
            bbox_inches = 'tight'
        )
        
        # Display the plot
        plt.show()
        
        # Close the plot to free up resources
        plt.close()
    
        return

    def plot_roc_curve(self, FPR, TPR):
        """
        Plot the Receiver Operating Characteristic (ROC) curve.
        Parameters:
        - FPR (list or array-like): False Positive Rate values.
        - TPR (list or array-like): True Positive Rate values.
        """
        
        # Create a new figure for the ROC curve
        plt.figure()
        
        # Plot the ROC curve using FPR and TPR with specified attributes
        plt.plot(FPR, TPR, lw = 2, color = 'blue', label = 'ROC Curve')
        
        # Set x and y axis limits, add grid, and remove top and right spines
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.grid(color = 'black', linestyle = '--', linewidth = '1')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        # Set labels for x and y axes, and add a title
        plt.xlabel('False Pos. Rate', fontsize = 12, weight = 'bold')
        plt.ylabel('True Pos. Rate', fontsize = 12, weight = 'bold')
        plt.title("ROC Curve", fontsize = 16, weight = 'bold')
        
        # Set font sizes for ticks, x and y labels
        plt.xticks(fontsize = 12)
        plt.yticks(fontsize = 12)
        
        # Save the plot as a PNG file and display it
        plt.savefig('roc.png', bbox_inches = 'tight')
        plt.show()
        
        # Close the figure to free up resources
        plt.close()
        
        return

    def compute_rates(self):
        FPR = []
        FNR = []
        TPR = []
        
        for threshold in self.thresholds:
            TP = 0
            FP = 0
            TN = 0
            FN = 0
            
            # Loop over genuine scores
            for genuine in self.genuine_scores:
                if genuine >= threshold:
                    TP += 1
                else:
                    FN += 1
            
            # Loop over impostor scores
            for impostor in self.impostor_scores:
                if impostor <= threshold:
                    TN += 1
                else:
                    FP += 1
            
            # Calculate the rates and append to their respective lists
            TPR_val = TP / (TP + FN + self.epsilon)
            FPR_val = FP / (FP + TN + self.epsilon)
            FNR_val = FN / (FN + TP + self.epsilon)
            
            TPR.append(TPR_val)
            FNR.append(FNR_val)
            FPR.append(FPR_val)
        
        return FPR, FNR, TPR

        
        
def create_graphs(gen,imp):
    # Replace these with your actual data (replace with your own dataset)
    genuine_scores = gen  # Replace with actual genuine scores (list or array)
    impostor_scores = imp  # Replace with actual impostor scores (list or array)
    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)
    # Print the number of genuine and impostor scores
    print(f"Number of genuine scores: {len(genuine_scores)}")
    print(f"Number of impostor scores: {len(impostor_scores)}")
    print("Genuine Score Range:", min(genuine_scores), max(genuine_scores))

    # Create a name for the system (you can keep it simple if only one set of scores)
    system_name = "Facial Hair Scores - 68 Landmarks"  # Replace with your system name

    # Create an instance of the Evaluator class with your actual scores
    evaluator = Evaluator(
        epsilon=1e-12,
        num_thresholds=200,
        genuine_scores=genuine_scores,
        impostor_scores=impostor_scores,
        plot_title=system_name
    )

    # Generate the FPR, FNR, and TPR using 200 threshold values equally spaced between -0.1 and 1.1
    FPR, FNR, TPR = evaluator.compute_rates()

    # Plot the score distribution. Include the d-prime value in the plot's title
    evaluator.plot_score_distribution()
    
    # Plot the DET curve and include the EER in the plot's title
    evaluator.plot_det_curve(FPR, FNR)
    
    # Plot the ROC curve
    evaluator.plot_roc_curve(FPR, TPR)

create_graphs(gen_scores,imp_scores)

