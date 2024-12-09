# Effects-of-Glasses-and-Facial-Hair-on-Facial-Recognition
Biometrics refers to using measurable physical characteristics or behavioral traits to identify or authenticate someone’s identity. These traits can range from fingerprints, facial features, gait, and many more identifiable aspects of a person. With these features, it is possible to create a biometric system which is a system that collects features and uses them to identify or authenticate users. One well-known identifiable feature of people is their face. Facial recognition is a major part of biometrics and is used in many mobile devices, such as smartphones. These devices collect facial features and use robust systems to compare the measurable aspects of a face against previously obtained templates. By providing a system with faces, it can create profiles for people. These profiles, or templates, can then be tested on with queries to see if properly recognizes people. With this knowledge, our group created a system that trained two models on a set of faces from “The Hong Kong Polytechnic University Disguise and Makeup Faces Database”. We focused on two questions during our process, which influenced how we trained the models: “How does the use of 5 facial landmarks compared to 68 facial landmarks impact the accuracy and robustness of Face ID systems when identifying individuals with varying amounts of facial hair?” and “How does varying the decision threshold influence the performance of Face ID systems in identifying individuals wearing glasses?” These questions were chosen because glasses and facial hair are very common and volatile occlusions that may appear on a person’s face. By testing glasses and facial hair, we can understand how much these occlusions might affect facial identification systems along with the impact of landmarks. To test these questions, we obtained the facial features of every available photo in the dataset. We then used a file containing the “ground truths” to obtain information on which pictures contained which specific features. For example, people wearing glasses would be labeled “0” or “1” to represent no glasses or glasses. Once we had the details on each image, we trained the classifiers on a dataset of people who did not have glasses or who did not have any facial hair. This was then tested on the dataset that had the trait to see how well the classifier would place each image. Genuine scores would represent scores for faces that were the same person query. Impostor scores would represent scores for people who were not the same person as the query. Our results showed that, in general, when classifiers are trained on human faces without occlusions, they struggle to create high genuine scores. However, they are still able to accurately generate low scores for impostors and high level of discrimination between genuine faces and impostors. Additionally, we observed a significant improvement in system performance when experimenting with 68 landmarks on faces with facial hair. On average, images with facial hair have higher genuine scores than images with glasses. Another thing we observed is that experimenting with the systems range of thresholds also improved performance, but not significantly.  
