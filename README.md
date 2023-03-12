# 18744-off-road

images - the folder containing the rgb images to be trained 
labels - the folder containing the masks that we’ll use for our training and validation, these images are 8-bit pixels after a colormap removal process.
results - the output of the trained model on the rgb images
results_color - the colored output of the trained model on the rgb images
model - the folder containing the code to run tasks
orig_colorLables - the folder containing the original groundtruth colored labels
orig_images - the folder containing the original rgb images (with groundtruth)
orig_images_unlabeled_Part01 - the folder containing the unlabeled images
orig_labels - the folder containing the original groundtruth labels
valid.txt - contains a list of images names randomly selected for validation
codes.txt - contains a list with classes names

# HOW-TO-RUN
we use four files to run the task
- train_without_weights.py
- train_with_weights.py
- predict.py
- color_results.py

Step1. run train_without_weights.py -> it will use images & labels folder to do supervised learning. Note here, you need to choose your learning rate based on the figure (loss_plot.png). After that, it will save the model in images/models/stage-2.pth

Step2. run train_with_weights.py -> it will use images & labels folder to do supervised learning. Note here, you need to choose your learning rate based on the figure (loss_plot_with_weights.png). After that, it will save the model in images/models/stage-2-weights.pth (this is the final model we will use)

Step3. run predict.py -> it will use images folder to do prediction on all these images, and save the results in results folder

Step4. run color_results.py -> it will use results folder to color the results/masks in results folder

*You need to change learning rate, the number of iterations yourself. They should have been commented with "TODO", so pay attention. Moreover, when do prediction & color_results, you can change the folder name in the files to predict and color on another folder.

# Semi-supervised-learning
After a round of supervised learning as mentioned above, we can run semi.py.
This will use the rgb images in the folder called orig_images_unlabeled_Part01 to do prediction. It will automatically select images with high confidence and save the rgb images in images folder and the corresponding/labels in the labels folder. Therefore, next time, one can directly follow the HOW-TO-RUN again but with pseudo-labeled data.

*You may want to change the way/threshold to select the images with high confidence, commented with "TODO" as well.