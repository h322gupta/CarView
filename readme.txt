
DataSet Prepration:

step1: Extracted polygon coordinates of all the present parts for a given image and calculated their area
step2: Then created the tsne plot for the area vectors
step3: Applied kmean clusteing on the output of step2 and created 6 clusters 
step4: Splited the data into 6 different folders and assigned the names by visual inspection 
step5: Filtered the anomal points by visual inspections

Model:

1. Transfer learning is used to train the model 
Why it is used:

1. VGG16 , Resnet or imagenet architecture are trained with millions of images and has proved their potential is the past
2. To train a deep learning model from scratch, huge dataset is required which is not available.

VGG16 model is finally selected as it gave relatively better performance (f1-score =~ 0.7) 


Model specific information:

1. VGG16 
2. Batch_size = 32
3. 2 fully connected layers 1 maxpooling layer 
4. First layer has 512 neurons with 50% dropout and second layer has 7 neuron each pertaining to one class and one for other
5. Number of epoch : 50

Limitation: This model has some issues in classifying left and right side images , it correctly classifies  the front and back more than 90% but for other classes this figure is relatively low.

With more data cleaning and training this issue can be resolved
