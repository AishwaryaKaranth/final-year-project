## About Source  

The Galaxy Zoo Challenge was started by Kaggle who teamed up with Winton Capital and Galaxy Zoo where participants are to classify galaxies into different categories. A large number of volunteers manually classified thousands of galaxies into various categories through the citizen science project. But manual classification is not feasible as there are billions of galaxies in our Universe. Therefore the primary aim is to develop an algorithm that mimics the manual classification or perform better than manual classifications.

The galaxies were manually classified based on a decision tree which consisted of 11 questions based on the answers to these questions the galaxies were classified into 37 different categories. For the Galaxy Zoo challenge, the dataset which was acquired by Sloan Digital Sky Survey(SDSS) was provided. It consisted about 64000 RGB images of galaxies. A corresponding training solutions file was provided which consisted of probability distribution of galaxies belonging to a particular class.

#### Decision Tree
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

### Final Model

The ninth model was run in three different runs instead of one single run. Each run had varying number of epochs. The first run had 6 epochs and gave training accuracy of 74.31% and validation accuracy of 74.31% as well. The training loss and validation loss for first run was 0.0122. The second run was for 7 epochs and it gave training accuracy was 77.7% and validation accuracy was 76%. The training loss for the second run was 0.0099 and validation loss was 0.0110. The third run was for 9 epochs and gave a training accuracy of 78.56% and validation accuracy of 77.18%. The training loss for third run was 0.0088 and validation loss was around 0.0102. This model which was trained for three different runs yielded a much better result than previous models which were trained in a single run. Since this model outperformed the previous models, it was also used to train unprocessed images. (***Accuracy to be mentioned***).

The final model was built using Keras libraries and built in functions. Once the architecture part is built as explained in the earlier section, we imported the training solutions csv file which contains the probability distributions for the classifications for each of the images which spans over 37 columns and a unique Galaxy ID. The Galaxy ID is used to map probability distribution with the images. These probability distributions are in the form of classes and can be considered as labels. 

`data=pd.read_csv('training_solutions_modified.csv')`

We then coded a function to map the image name and the galaxy ID. 

`def apply_jpg(f):`

`return f+ ".jpg"` 

To achieve this, we use the `apply()` method which takes a function and applies to each image in the pandas dataframe. 

`data["GalaxyID"]=data["GalaxyID"].astype(str).apply(apply_jpg)`

Here, `apply()` takes in `apply_jpg` function and appends ***.jpg*** extension to each image in the dataframe. :

We then split the dataset into training set and validation set in 80:20 ratio. 20% of the dataset was used for validation. We used keras built in function ImageDataGenerator() to acheive the same.

`datagenerator=data.ImageDataGenerator(validation_split=0.2)` where `data` is the training solutions csv file.

We then use the built-in keras function called `flow_from_dataframe()` to automatically map the *GalaxyID* column with the images. 

`train_generator = datagenerator.flow_from_dataframe(
    dataframe=data,
    directory="/content/gdrive/MyDrive/Processed/Data/he1/",
    x_col="GalaxyID",
    y_col=classes,
    subset="training",
    batch_size=64,
    shuffle=True,
    class_mode="raw",
    target_size=(224, 224))`

The first parameter of `flow_form_dataframe()` is `dataframe`, which is the training solutions csv file that we imported earlier. The second parameter is the directory, it specifies the directory of the images stored. The next parameter is `x_col`, which indicates the filenames of the images in the csv file which in this case is the GalaxyID, `x_col='GalaxyID'`. The next parameter is `y_col` indicates the labels. We put all 37 classes ina list called classes therefore `y_col=classes`. The next parameter is subset which indicates which part of dataset we are dealing with. For the `train_generator` the subset is training, that is the training set. The next parameter is `batch_size` which we have chosen as 64. To increase randomness we have set `shuffle=True`. The next parameter is class_mode, which is the mode to yield targets. Here we have set `class_mode` to `raw` as the targets in our case are probability distributions which are numerical values. The next parameter is `target_size`, which indicates the size of the image. The default size that we have chosen is 224* 224.

The same procedure is used for validation. The parameters remain the same.

`valid_generator = datagenerator.flow_from_dataframe(
    dataframe=data,
    directory="/content/gdrive/MyDrive/Processed/Data/he1/",
    x_col="GalaxyID",
    y_col=classes,
    subset="validation",
    batch_size=64,
    shuffle=True,
    class_mode="raw",
    target_size=(224, 224))`

We then chose the training step size and validation step size. The number of samples in training set was divided by the batch size to obtain training step size. The number of samples in validation set was divided by the batch size to set validation step size. 

`train_step_size = train_generator.n // train_generator.batch_size`

`train_generator.n` returns the number of images in training set.

`train_generator.batch_size` returns the batch_size that we had set to 64.

The number of samples in validation set was divided by the batch size to set validation step size. 
`validation_step_size = valid_generator.n // valid_generator.batch_size`

`validation_generator.n` returns the number of images in validation set.

`validation_generator.batch_size` returns batch_size which is set to 64.

Running the above code snippet gave 197050 validated training image filenames and 49262 validated validation image filenames. The training step size was 3078 and validation step size was 769.

We then proceeded to fit the model. WE imported hdf5 module to save and load weights. 

We trained the model in three separate runs. Each run had varying number of epochs. We used to callbacks to automate certain tasks after every epoch, it also lets us control the training process. Callbacks include stopping the trianing process when the accuracy or other concerned metric does not increase after certain number of epochs, saving the model after each epoch, loading previous weights and so on. We used early stopping to prevent the model from running if the validation accuracy does not improve after a certain number of epochs. Early stopping is primarily used to prevent overfitting. Early Stopping takes in a number of parameters. 

`earlystopping = callbacks.EarlyStopping(monitor ="val_loss",
                                        mode ="auto", patience = 3, verbose=2,
                                        restore_best_weights =True)`

`monitor` is used to monitor the concerned metric. Here we are monitoring validation loss.

`patience` indicates number of epochs after which if the validation accuracy or any other concerned metric does not improve the training process stops.

`verbose` is mostly used for formatting the output.

`restore_best_weights` restores only the best weights of the model when the training process stops when set to `True`

We then made use of ModelCheckpoint to save the model after every epoch. 

`checkpoint = callbacks.ModelCheckpoint(filepath='/thirdrunweights.hdf5', verbose=2, save_best_only=True)`

The parameters for `ModelCheckpoint` are `filepath` which takes in the filepath to store the weights after each epoch. `verbose` is here used to format the output. We also stored only the best weights after the training process stops therefore we set `save_best_only` parameter to `True`.

We finally fit the model using model.fit(). model.fit() takes in several parameters. It takes in training data which in out=r case is train_generator. It takes in the training step size, number of epochs for which the model should run, validation data , validation step size and callbacks. Here `callbacks` is a list containing `early_stopping` and `ModelCheckpoint`.

`model.fit(train_generator,
           steps_per_epoch=train_step_size,
           epochs=20,
           validation_data=valid_generator,
           validation_steps=validation_step_size,
           callbacks =[checkpoint,earlystopping],
           verbose=2 )`

Here training data is train_generator that was explained earlier, `steps_per_epoch` is the training step size which was set to 3079. The model was run in three different runs and each of those runs had different number of epochs. At each run the previous weights were restored and run. The first run had 6 epochs, second run had 7 epochs and third run had 9 runs. `validation_data` is `validation_generator` that was defined earlier. `validation_steps` is validation step size which was set to 769. Callbacks list contained checkpoint, early stopping. Verbose again was used to format the output data.

We also used `timeit` module to determine how long the model took to run.

The model was run with a learning rate 0.001 and decay 5*10^-4 with a batch size of 64. The first run 6 epochs. ***(Insert loss plot 1)*** The total number of epochs was 6. The loss plot has epochs as x-axis and loss as y-axis. As the the number of epochs gradually increased the loss gradually decreased. The validation loss at the end of first run was 0.0122. ***(Insert loss plot 2)*** The loss decreased slightly as the epochs increased but validation loss became more constant towards the end of of 6th epoch. The validation loss at the end of second run was 0.0110. ***(Insert loss plot 3)*** The total number of epochs in the third run was 9. The loss decreased initially as the epochs increased but later remained mostly constant and started decreasing slightly at the end of 9th epoch which indicated overfitting therefore we stopped training after 9 epochs in the third run. The validation loss at the end of third run was 0.0102.
During the first run, ***(Insert accuracy plot 1)*** the accuracy increased as the number of epochs increased. The validation accuracy at the end of first run was 74.31%. ***(Insert accuracy plot 2)*** The numbers of epochs was 7 and accuracy significantly increased until 4 epochs and then gradually increased as the number of epochs increased. The validation accuracy at the end of second run was 76%. ***(Insert third accuracy plot)*** The number of epochs was 9 and validation accuracy slightly increased during the first two epochs and then remained constant as the epochs increased. The validation accuracy at the end of the third run was 77.18%.
- Aishwarya

## 
<hr>

## Image pre-processing
Data of any form cannot directly be used for any model training techniques,they have to be subjected to certaining conditioning before being  fed into an algorithm.Images may be overlapped with noise,they may have poor contrast,can be of varied sizes and certain algorithms use a standards for image sizes being fed to them.When the amount of the data available is not sufficient,certain techniques must be employed to prevent overfitting the model,concepts like data augmentation is widely used for reducing overfitting.
We have employed certain Pre-Processing techniques on the images .We mainly used the PIL(Python Imaging Library)and OpenCv Libraries to implement the following processes.Both the libraries have pre-defined functions for carrying out basic image processing operations.
### Resizing
The image dataset that we obtained from kaggle challenge had images of size 424x424.
A standard VGG-16 model uses images of size 224x224.Also larger the size of an image greater is the computing involved therefore resizing the images to certain standards could reduce the computing power required.We therefore resized the images using a python library called PIL.  
### Data augmentation:
The main concerns of any model training is the problem of overfitting.The usual techniques of overcoming overfitting is using certain regularization techniques Data augmentation is one of them.It involves generating samples artificially ,thereby increasing the size of the dataset.Images are usually subjected to certain transformation processes like random rotations,flipping etc
We employed the technique of rotating the images at random angles thereby increasing the size of our dataset from 61578 to 246312 where each image was subjected to rotation in 4 random angles
We used PIL library to implement the same.
#### Implementation steps
* Read an image
* Generate a random number between 1 and 360 using `random.uniform()`, a method to randomly generate floating numbers inclusive of both the parameters provided
* Use the randomly generated number from the above step as an argument to the resize function.
* Perform the same process 4 times for each image

### Median Filtering
The images available might be overlapped with noise, hence it is essential to use certain filtering techniques to filter them. 
Median filtering was preferred since it has negligible loss of edges, which we want to preserve for our model training.The filter has a slight smoothing effect on our images.
We again employed the PIL library where ImageFilter module was used which has certain predefined filters that is used with the `filter` function with the kernel size set to 3.
### CLAHE
The images we obtained had poor contrast,Histogram Equalization is generally used for increasing contrast in images,but we used CLAHE (Contrast Limited Adaptive Histogram Equalization), a variant of histogram equalization technique where it keeps a check on the amplification of contrast.A regular Histogram Equalization technique that concentrates on global contrast sometimes does not yield better results.
CLAHE algorithm proceeds with considering small patches of the image(subimages) called “tiles” also preventing overamplification as overamplification may result in adding noise to the image.
Open Cv was used to implement CLAHE.
#### Implementation steps:
* We decided to apply CLAHE only on the Lightness channel of the image.
This required us to convert the RGB image to LAB using `cv2.cvtCOLOR()` function provided by the Open CV library.(Lab color space is a 3-axis color system with dimension L for lightness and a and b for the color dimensions.)
The Lab image was then split into three channels using the `split()` function.
* Creating titles of size 3x3 and setting the clip-limit to 2 and passing these as parameters to  `createCLAHE()`.Clip limit sets the threshold for contrast amplification.
* Then calling the `apply()` function on the object to apply the equalization on the L channel of the image
* The image channels are then merged using the `merge` function
* Convert the images back to RGB colour space.
<hr>
