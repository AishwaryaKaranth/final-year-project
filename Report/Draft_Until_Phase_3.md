## About Source  

The Galaxy Zoo Challenge was started by Kaggle who teamed up with Winton Capital and Galaxy Zoo where participants are to classify galaxies into different categories. A large number of volunteers manually classified thousands of galaxies into various categories through the citizen science project. But manual classification is not feasible as there are billions of galaxies in our Universe. Therefore the primary aim is to develop an algorithm that mimics the manual classification or perform better than manual classifications.

The galaxies were manually classified based on a decision tree which consisted of 11 questions based on the answers to these questions the galaxies were classified into 37 different categories. For the Galaxy Zoo challenge, the dataset which was acquired by Sloan Digital Sky Survey(SDSS) was provided. It consisted about 64000 RGB images of galaxies. A corresponding training solutions file was provided which consisted of probability distribution of galaxies belonging to a particular class.

##### Decision Tree
The decision tree that was used to classify the galaxies consisted of 11 questions based on which they were classified into 37 different categories.

The first question asks if the galaxy is smooth and rounded, with a sign of a disk. It has three options— smooth, features or disk, star or artifact. If the galaxy is smooth, then the tree checks how rounded the galaxy is, if it is completely round, if it is in between or cigar shaped. The tree then goes on to check if there is any odd feature in this smooth galaxy or if this galaxy is distributed or irregular. The odd feature could be a ring, lens or arc, dust lane, irregular etc. The decision tree for the smooth galaxy ends here.

If the answer to the first question of the decision tree that is if the galaxy is smooth, rounded with no sign of a disk is ***features or disk visible in galaxy***, then the next question asked is if there disk can be viewed edge-on. If the disk can be viewed edge-on, it checks if the galaxy has a bulge at its center and it also checks the shape of the bulge. The bulge could be rounded, boxy or there could be no bulge. After checking for bulge, it checks for odd features and if there are odd features it checks what those odd features are. But if the galaxies has features or disk and is not viewed edge-on, it checks if there is bar feature through the center of the galaxy. If there is a bar feature through the center of the galaxy, it checks for spiral patterns. If there are signs of spiral arm patterns, it checks how the spiral arms are wound, if they are tightly wound, loosely wound or medium wound and then checks how many spiral arm patterns there are. After determining the number of spiral arms, it checks how prominent the central bulge is. The bulge could be dominant or barely noticeable or it there might not be any bulge. After checking about the prominence of the bulge, it checks for odd features and what they are if they are present. If the galaxy has features or disk and cannot be viewed edge on and does not have any spiral arm patterns visible, the decision tree checks for the prominence of the bulge and then checks for the odd features if there are any. The decision tree ends here for galaxies that have features or disk.

If the answer to the question is the galaxy smooth, rounded with no sign of disk is ***star or artifact*** then the decision tree doesn't check any other questions further and ends here as it is not a galaxy.
<br />
- Aishwarya

## 
