# Classification under Changes in Sequences Order

# Datasets

We used five grayscale image datasets, all of them with images of 28×28 pixels:
(a) The MNIST is a dataset of handwritten digits (0–9), that has 10 classes, with 60,000 images for training and 10,000 for testing. (b) The MNIST-C dataset is a comprehensive suite of 15 corruptions applied to the MNIST dataset.We choosed the MNIST-C glass blur corruption from 15 corruptions sorts as our second dataset. It has 60,000 images for training, 10,000 for testing, and 10 classes as the MNIST. (c) The notMNIST dataset contains images of the English letters from A to J, thus it consists 10 classes. The original training set contains 500,000 images and the test part 19,000, from these we selected randomly [15] 60,000 for training and 10,000 for testing. (d) The FashionMNIST dataset has 10 classes as well, and consist on images of Zalando’s articles from T-Shirt, Trouser till Bag, and Ankle boots. It has 60,000 and 10,000 images for training and testing, respectively. (e) The Sign Language MNIST (SLMNIST) dataset contains all the alphabet, except J and Z, of the standard American Sign Language (ASL). It consists of 27,455 and 7,172 images for training and testing, respectively, with 24 classes.

# The aim of the project

We are interested in understanding if the order within the sequences, and their length, have an effect on the overall performance of the RNNs, particularly when they are based on LSTM units. This question maybe relevant for data setup procedures, and it may be useful for video tagging and for general image description.

# Experiment design

With the purpose of selecting the appropriate LSTM architecture and parameter, to classify each image-datasets under the same conditions, first we tested different configurations. The best performance was achieved using the following parameters: three layers, 512 LSTM units per layer, a learning rate of 0.001 with exponential decay, and 256 images per batch size. Also, an early stopping condition has been implemented, consisting on training until there are no changes in
the loss function. The model is compiled with Adam optimizer, the crossentropy loss function and metric accuracy.

Various training datasets were built for each image dataset, following three different rules. The first rule considers to build the sequences from the rows of the images, which we call horizontal order (H); the second rule consists of building
the sequences using the columns of the images, which we call vertical order (V); and the third rule, that we call spiral order (S), consists of building the sequences by collecting the pixels from the center and going out from the center in circular ways till the border of the image.

A grayscale image in X commonly comes with a M×N pixels, in this work all raw images are 28×28 pixels. From the model’s point of view, varying the shape M × N of an image is equivalent to varying the sequences length and using the sorts of order (H, V and S) is equivalent to varying the order in each sequence. M defines the number of sequences and N defines the length of a sequence that can be got in an image. 

<img width="250" alt="image" src="https://github.com/ekchacon/Classification-under-Changes-in-Sequences-Order/assets/46211304/25c4099d-8d2a-43f1-b05d-c57468f3c6b8">

An experiment E is carried on with a specific combination of M × N shape of the image and a sort order H, V or S as shown by the previous figure. Each experiment yields an accuracy.

The M × N combinations considered in this work are as follow: (2,392), (4,196), (7,112), (8,98), (14,56), (16,49), (28,28), (49,16), (56,14), (98,8), (112,7) and (196,4). These shapes were chosen mainly because they have an integer value
for M and N. As previously mentioned, the datasets used in this work are the well known MNIST, MNIST-C, notMNIST, FashionMNIST and Sign Language MNIST. The combinations of datasets, image shapes and order sorts yield a total of 180 experiments.
