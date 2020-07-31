# Hand Tracking Pytorch
 Hand tracking application developed in Pytorch.

Hand tracking application developed for an university project. It used a [RCNN](https://arxiv.org/abs/1311.2524) (Regions with CNN) to track hands inside the image and a CNN (Convolutional Neural Network) to evaluate the number of the fingers in the hand.

The RCNN training script is based on the script at [tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html) in the Pytorch site. I use a pretrained RCNN on COCO dataset and with fine tuning I teach the network how to recognize hands. The original tutorial used a Mask-RCNN that also create a mask of the shape of the things you search. Since I was only interested in a square box aruond hand I modify the script to use a *simpler* RCNN.

Since the big dimension of the RCNN I can't saved the trained model in github but I upload it. You can found it at this [link](https://drive.google.com/file/d/1bmWjiUp1Lq9ggnsOGAugxVz4GROcR2VO/view?usp=sharing). The CNN model is lighet so I manage to upload it on github.

To test the track in real time you also need [OpenCV](https://opencv.org/).

## RCNN Training
I try two different dataset to train the network. The firs is the pretty famous [EgoHands Dataset](http://vision.soic.indiana.edu/projects/egohands/) (The 1.3 Gb version) and the other is a my personal (costum) dataset that I create. The results are similar and you could choose which one to use. The trained model provided in the link is trained on my personal dataset. 

### Costum dataset use and creation (*RCNN (my dataset)* folder)
To create your own dataset used the file inside the *RCNN (my dataset)* folder. The files are well commented but basically you need to:
1. Execute the *dataset_creator.py* (you need OpenCV for this). This script create a windows with the image capture by your camera and n boxes (n set by you). You position the hand inside the boxes and after x seconds (x decide by you) the image and the boxes position are saved.
2. Check the dataset with *dataset_check.py* (OPTIONAL).
3. Train the network with *train_RCNN.py* .

The *MyDataset.py* file contain the Dataset class used during training. This class is an extension of the Pytorch Dataset class and it will used in combination with the Dataloader (more info and example [here](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html), [here](https://pytorch.org/docs/stable/data.html), [here](https://pytorch.org/docs/stable/torchvision/datasets.html) and [here](https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel)).

The *support_function.py* contain function used during the dataset creation and the dataset loading.

**NB:** The train need torchvision. All the file are inside the *vision* folder and that folder must be in the same folder of *train_RCNN.py*.

### EgoHands dataset use (*RCNN (egos hand dataset)* folder)
To train the model with the EgoHands dataset download the dataset from the [link](http://vision.soic.indiana.edu/projects/egohands/) (The 1.3 Gb version). Extract it and put the folders inside the *_LABELLED_SAMPLES* in tha path that you will use to contain the file for the training. After that execute the file *train_RCNN.py* .

**NB:** The train need torchvision. All the file are inside the *vision* folder and that folder must be in the same folder of *train_RCNN.py*.

## CNN Training (*CNN (Finger Counter)* folder)
The CNN is a simple network trained on a costum dataset. The dataset is created with the file *dataset_creation.py* in a similar way to the costum dataset for the RCNN. I use an Adadelta optimizer with a Cross Entropy loss function. For some reason sometimes the training succed and sometimes remain stuck and don't learn after hundreds of epoch. In the last case stop and restart the training. 

I use 140 x 140 square as size of the image but you could choose the size that you prefer.

The trained network provided with the code doesn't work very well since I train its on 200 example. Also I suggest you to use always the same hand to improve the accurancy. A trick later used in the the final script is to mirror one of the two hand find by RCNN so the classifier will only receive *one type* of hand and give similar results for both hand.


## Hand tracking script
The *hand_detector_V2.py* contain the script that track the hand(s). It use OpenCV to capture the camera image and feed that image to the RCNN. With the RCNN I find the hand, take only that portion of the image, reshape in a 140 x 140 square and feed them into the fingers counter.

**NB:** You need a GPU with CUDA to obtain a (more or less) smoot frame rate. I use a GTX 2070 but probably even less powerfull GPU work. To improve the FPS I use a little trick: instead of feed every frame into the RCNN I do it every 3 frame. In this way the code works smoothly granted that you had a GPU capable of executing the code require by the RCNN in relative small time.

## TODO List
- [ ] Add video explanation.
- [ ] Modify the costum dataset to label differently left hand and right and.
- [ ] Retrain the RCNN to allow recognition of left and right hand.

## Contacs 
If you ever use this code please cite me  :pray:  :heart:

If you need to contact my you could write me at *alberto.zancanaro.1@gmail.com* or *alberto.zancanaro.1@studenti.unipd.it* .

