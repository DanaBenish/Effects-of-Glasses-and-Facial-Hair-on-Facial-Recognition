# Biometrics and Facial Recognition

Biometrics refers to using measurable physical characteristics or behavioral traits to identify or authenticate someone’s identity. Common biometric traits include fingerprints, facial features, gait, and more. One of the most widely used forms of biometrics is facial recognition, which is implemented in devices like smartphones to authenticate users. Facial recognition systems work by collecting and comparing measurable aspects of a person’s face to pre-existing templates. These templates, created from images, help the system identify or verify individuals by comparing the features of their face.

Project Overview:
In this project, we focused on the impact of facial hair and glasses on facial recognition accuracy, using "The Hong Kong Polytechnic University Disguise and Makeup Faces Database". We trained two models with different sets of facial landmarks (5 vs. 68 landmarks) to explore the effects of these common facial occlusions on Face ID systems.

Research Questions:
How does the use of 5 facial landmarks compared to 68 landmarks impact the accuracy and robustness of Face ID systems when identifying individuals with varying amounts of facial hair?
How does varying the decision threshold influence the performance of Face ID systems in identifying individuals wearing glasses?
These questions were chosen because facial hair and glasses are common, yet challenging, features that can interfere with facial recognition systems. By analyzing these factors, we aim to understand their impact on the accuracy of biometric systems.

Approach:
We utilized the dataset to extract facial features from images, labeling specific traits such as glasses and facial hair.
Using a ground truth file, we classified images based on these traits (e.g., images labeled "1" for glasses and "0" for no glasses).
We trained classifiers on datasets containing people without these traits and then tested them on images with glasses and/or facial hair to assess the model’s performance.

Key Files:
Facial Hair and Glasses Classifiers: Scripts for training and testing models.
Ground Truth Data: Labels indicating presence of glasses or facial hair in images.
