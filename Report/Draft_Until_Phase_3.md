## About Source  

The Galaxy Zoo Challenge was started by Kaggle who teamed up with Winton Capital and Galaxy Zoo where participants are to classify galaxies into different categories. A large number of volunteers manually classified thousands of galaxies into various categories through the citizen science project. But manual classification is not feasible as there are billions of galaxies in our Universe. Therefore the primary aim is to develop an algorithm that mimics the manual classification or perform better than manual classifications.

The galaxies were manually classified based on a decision tree which consisted of 11 questions based on the answers to these questions the galaxies were classified into 37 different categories. For the Galaxy Zoo challenge, the dataset which was acquired by Sloan Digital Sky Survey(SDSS) was provided. It consisted about 64000 RGB images of galaxies. A corresponding training solutions file was provided which consisted of probability distribution of galaxies belonging to a particular class.

##### Decision Tree
The decision tree that was used to classify the galaxies consisted of 11 questions based on which they were classified into 37 different categories.

The first question asks if the galaxy is smooth and rounded, with a sign of a disk. It has three options— smooth, features or disk, star or artifact. If the galaxy is smooth, then the tree checks how rounded the galaxy is, if it is completely round, if it is in between or cigar shaped. The tree then goes on to check if there is any odd feature in this smooth galaxy or if this galaxy is distributed or irregular. The odd feature could be a ring, lens or arc, dust lane, irregular etc. The decision tree for the smooth galaxy ends here.

If the answer to the first question of the decision tree that is if the galaxy is smooth, rounded with no sign of a disk is ***features or disk visible in galaxy***, then the next question asked is if there disk can be viewed edge-on. If the disk can be viewed edge-on, it checks if the galaxy has a bulge at its center and it also checks the shape of the bulge. The bulge could be rounded, boxy or there could be no bulge. After checking for bulge, it checks for odd features and if there are odd features it checks what those odd features are. But if the galaxies has features or disk and is not viewed edge-on, it checks if there is bar feature through the center of the galaxy. If there is a bar feature through the center of the galaxy, it checks for spiral patterns. If there are signs of spiral arm patterns, it checks how the spiral arms are wound, if they are tightly wound, loosely wound or medium wound and then checks how many spiral arm patterns there are. After determining the number of spiral arms, it checks how prominent the central bulge is. The bulge could be dominant or barely noticeable or it there might not be any bulge. After checking about the prominence of the bulge, it checks for odd features and what they are if they are present. If the galaxy has features or disk and cannot be viewed edge on and does not have any spiral arm patterns visible, the decision tree checks for the prominence of the bulge and then checks for the odd features if there are any. The decision tree ends here for galaxies that have features or disk.

If the answer to the question "is the galaxy smooth, rounded with no sign of disk" is ***star or artifact*** then the decision tree doesn't check any other questions further and ends here as it is not a galaxy.
<br />

## Comparison of different models trained

We trained a total of 9 models with varying epochs, learning rate, optimizers and decay to determine which model gave the best result. Our focus was on reducing validation loss than on achieving higher accuracy since the problem is more of a multi-regression problem than a standard classification problem.

The first model that we trained was on unprocessed images. The unprocessed images were obtained after performing resizing, rotation and augmentation on original images from the dataset. We used RMSProp as optimizer with a learning rate 10^-6 and batch size 64. It was run for a total of 45 epochs. It yielded a training accuracy of 67.19% and a validation accuracy 67.98%. The training and validation loss in this case were the same that is 0.0162. 

We trained a second model on unprocessed images. We used RMSProp as optimizer but we changed the learning rate to 10^-4 and batch size to 32. It was run for a total of 50 epochs. It gave training accuracy of 82.22% and a validation accuracy of 75.46%. The training loss was around 0.0203 and validation loss was 0.0187. It performed better than the first model.

The third model was trained on processed images. Median filters and histogram equilization was done on unprocessed images to obtain processed images. We used RMSProp optimizer for this model. The learning rate was 10^-4 and batch size was chosen to be 32. It was run for 50 epochs and yielded a training accuracy of 81.87% and validation accuracy of 75.46%. The training loss was 0.0062 and validation loss was 0.0119.

The fourth model was trained on original images from the dataset. The original images were not processed. They were resized, rotated, augmented by using the keras built-in function called `ImageDataGenerator()` . The images were resized to 224 * 224 and were rotated and augmented using `ImageDataGenerator()`. The idea was to find out if the built-in function for data augmentation, rotation and resizing performed any better than the functions that we wrote from scratch to perform the data augmentation. It used Adam optimizer with a learning rate of 0.001 and the batch size was chosen to be 64. Decay was set to 5* 10^-4. It ran for 12 epochs and gave a training accuracy of 59.97% and a validation accuracy of 61.96%. The training loss was 0.0203 and validation loss was 0.0187.

The fifth model was also trained on original dataset images. The resizing, rotaation and data augmentation was done using `ImageDataGenerator()` , built-in keras function. We used Adam optimizer for this model. The learning rate was 0.001 and batch size was chosen to be 64. It was run for 15 epochs and yielded a training accuracy of 65.51% and validation accuracy of 62.78%. The training loss was 0.0119 and validation loss was 0.0172.

The sixth model was trained on processed images. We used Adam optimizer for this model. The learning rate was $0.001$ and batch size was chosen to be 64. It was run for 10 epochs and yielded a training accuracy of 72.96% and validation accuracy of 72.65%. The training loss was 0.0126 and validation loss was 0.0133.

The seventh model was trained on processed images. We used Adam optimizer for this model. The learning rate was 3* 10^-4 and batch size was chosen to be 64. It was run for 10 epochs and yielded a training accuracy of 77.15% and validation accuracy of 75.94%. The training loss was 0.0101 and validation loss was 0.0111.

The eighth model was trained on processed images. We used Adam optimizer for this model. The learning rate was $0.001$ and batch size was chosen to be 64. It was run for 13 epochs and yielded a training accuracy of 75.77% and validation accuracy of 74.54%. The training loss was 0.0108 and validation loss was 0.0120.

The ninth model was trained on processed images initially. Median filters and histogram equilization was done on unprocessed images to obtain processed images. We used Adam optimizer for this model. The learning rate was 0.001 and batch size was chosen to be 64. It was run in three different runs instead of running the entire model in one single run. It yielded a training accuracy of 78.56% and validation accuracy of 77.18% at the end of third run. the training loss was 0.0088 and the validation loss was 0.0102.
(Will add the tables in LaTeX).
- Aishwarya

## 
