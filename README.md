# Hand Tracking Pytorch
 Hand tracking application developed in Pytorch.

Hand tracking application developed for an university project. It used a [RCNN](https://arxiv.org/abs/1311.2524) (Regions with CNN) to track hands inside the image and a CNN (Convolutional Neural Network) to evaluate the number of the fingers in the hand.

The RCNN training script is based on the script at [tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html) in the Pytorch site. I use a pretrained RCNN on COCO dataset and with fine tuning I teach the network how to recognize hands. The original tutorial used a Mask-RCNN that also create a mask of the shape of the things you search. Since I was only interested in a square box aruond hand I modify the script to use a *simpler* RCNN.

Since the big dimension of the RCNN I can't saved the trained model in github but I upload it. You can found it at this [link](https://drive.google.com/file/d/1bmWjiUp1Lq9ggnsOGAugxVz4GROcR2VO/view?usp=sharing). The CNN model is lighet so I manage to upload it on github.

To test the track in real time you also need [OpenCV](https://opencv.org/).

## RCNN Training
I try two different dataset to train the network. The firs is the pretty famous [EgoHands Dataset](http://vision.soic.indiana.edu/projects/egohands/) (The 1.3 Gb version) and the other is a my personal (costum) dataset that I create. The results are similar and you could choose which one to use. The trained model provided in the link is trained on my personal dataset. 

### Costum dataset creation
