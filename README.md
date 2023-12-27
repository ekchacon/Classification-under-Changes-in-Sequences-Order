# Classification under Changes in Sequences Order

# The aim of the project

We are interested in understanding if the order within the sequences, and their length, have an effect on the overall performance of the RNNs, particularly when they are based on LSTM units. This question maybe relevant for data setup procedures, and it may be useful for video tagging and for general image description.

# Datasets

We used five grayscale image datasets, all of them with images of 28×28 pixels:
(a) The MNIST is a dataset of handwritten digits (0–9), that has 10 classes, with 60,000 images for training and 10,000 for testing. (b) The MNIST-C dataset is a comprehensive suite of 15 corruptions applied to the MNIST dataset.We choosed the MNIST-C glass blur corruption from 15 corruptions sorts as our second dataset. It has 60,000 images for training, 10,000 for testing, and 10 classes as the MNIST. (c) The notMNIST dataset contains images of the English letters from A to J, thus it consists 10 classes. The original training set contains 500,000 images and the test part 19,000, from these we selected randomly [15] 60,000 for training and 10,000 for testing. (d) The FashionMNIST dataset has 10 classes as well, and consist on images of Zalando’s articles from T-Shirt, Trouser till Bag, and Ankle boots. It has 60,000 and 10,000 images for training and testing, respectively. (e) The Sign Language MNIST (SLMNIST) dataset contains all the alphabet, except J and Z, of the standard American Sign Language (ASL). It consists of 27,455 and 7,172 images for training and testing, respectively, with 24 classes.

# Experiment design

With the purpose of selecting the appropriate LSTM architecture and parameter, to classify each image-datasets under the same conditions, first we tested different configurations. The best performance was achieved using the following parameters: three layers, 512 LSTM units per layer, a learning rate of 0.001 with exponential decay, and 256 images per batch size. Also, an early stopping condition has been implemented, consisting on training until there are no changes in
the loss function. The model is compiled with Adam optimizer, the crossentropy loss function and metric accuracy.

Various training datasets were built for each image dataset, following three different rules. The first rule considers to build the sequences from the rows of the images, which we call horizontal order (H); the second rule consists of building
the sequences using the columns of the images, which we call vertical order (V); and the third rule, that we call spiral order (S), consists of building the sequences by collecting the pixels from the center and going out from the center in circular ways till the border of the image.

A grayscale image in X commonly comes with a M×N pixels, in this work all raw images are 28×28 pixels. From the model’s point of view, varying the shape M × N of an image is equivalent to varying the sequences length and using the sorts of order (H, V and S) is equivalent to varying the order in each sequence. M defines the number of sequences and N defines the length of a sequence that can be got in an image. 

<img width="250" alt="image" src="https://github.com/ekchacon/Classification-under-Changes-in-Sequences-Order/assets/46211304/25c4099d-8d2a-43f1-b05d-c57468f3c6b8">

An experiment E is carried out with a specific combination of M × N shape of the image and a sort order H, V or S as shown by the previous figure. Each experiment yields an accuracy.

The M × N combinations considered in this work are as follow: (2,392), (4,196), (7,112), (8,98), (14,56), (16,49), (28,28), (49,16), (56,14), (98,8), (112,7) and (196,4). These shapes were chosen mainly because they have an integer value
for M and N. As previously mentioned, the datasets used in this work are the well known MNIST, MNIST-C, notMNIST, FashionMNIST and Sign Language MNIST. The combinations of datasets, image shapes and order sorts yield a total of 180 experiments.

# Results and discussion

Using the best architecture, we evaluated our experiments with the five datasets, organized in different shapes and order of pixels (H, V and S). Our main results are shown in the next table, from which we observe the following:

- Doubtless, the shape (28,28) or sequences with 28 features obtained the maximum accuracies within six experiments, equally distributed between the vertical and horizontal order.
- The shape (2,392) is the second with four maximum accuracies, mostly distributed in the spiral order.
- Generally, the horizontal and vertical order accuracies are slightly above the spiral order accuracies.
- Overall, we observe the maximum accuracies tend to stay above (28,28), where the sequences are longer than those below.

<img width="550" alt="image" src="https://github.com/ekchacon/Classification-under-Changes-in-Sequences-Order/assets/46211304/880412de-5289-4c36-85d6-e5f409b516f5">

We analyze the accuracies shown in the table  with the help of a boxplot chart, as shown in the next figure. In according to the median and maximum values of the boxplots, we can see the two first datasets (MNIST and MNIST-C) in the H order has better accuracies than the V and S orders. Also, the S order gets the lower accuracies in general. The same pattern appears in the FASHION dataset, though it has a smaller accuracy than MNIST and MNIST-C. 

The notMNIST and SLMNIST datasets, in the boxplot figure, have different results, the V order has a slightly better median accuracy than the H order median accuracy. Although one dataset is around 95% and the other around 85%, they both follow the same pattern. And the S order has the worst median accuracy in both cases. 

On the other hand, we can observe from the boxplot figure, accuracies in the H order are in general less disperse, and the S order has more dispersed accuracies. It means that working with horizontal order is more reliable than the others and it has better median accuracies and are less spread.

<img width="550" alt="image" src="https://github.com/ekchacon/Classification-under-Changes-in-Sequences-Order/assets/46211304/dc328fe3-8b49-43b8-baee-5dc8b6a84a59">

# Conclusion

In this work we presented a procedure to evaluate the accuracy of a recurrent neural network based on LSTM units, for datasets organized in sequences of different order and lengths. The experiments where conducted on five datasets of grayscale images of 28×28 pixels, where sequences were extracted from the images using three different rules. The results show that in most of the cases the best accuracies are achieved when the sequences are extracted from images following a horizontal order, but there are some cases where a vertical order can be better.
