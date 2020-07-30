# Hand Tracking Pytorch
 Hand tracking application developed in Pytorch.

Hand tracking application developed for an university project. It used a [RCNN](https://arxiv.org/abs/1311.2524) to track hands inside the image and a CNN to evaluate the number of the fingers in the hand.

The RCNN training script is based on the script at [tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html) in the Pytorch site. I use a pretrained RCNN on COCO dataset and with fine tuning I teach the network how to recognize hands. The original tutorial used a Mask-RCNN that also create a mask of the shape of the things you search. Since I was only interested in a square box aruond hand I modify the script to use a *simpler* RCNN
