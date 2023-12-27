# Classification under Changes in Sequences Order

# The aim of the project

<!-- This content will not appear in the rendered Markdown -->

<!-- Non-scientific: We are interested in understanding if the order within the sequences, and their length, have an effect on the overall performance of the RNNs, particularly when they are based on LSTM units. This question maybe relevant for data setup procedures, and it may be useful for video tagging and for general image description. -->

We aim to investigate the impact of sequence order and length on the overall performance of Recurrent Neural Networks (RNNs), especially those utilizing Long Short-Term Memory (LSTM) units. Addressing this query is pertinent for refining data setup procedures, with potential applications in video tagging and general image description.

# Datasets

We used five grayscale image datasets, all of them with images of 28×28 pixels: (a) The MNIST is a dataset of handwritten digits (0–9), that has 10 classes, with 60,000 images for training and 10,000 for testing. (b) The MNIST-C dataset is a comprehensive suite of 15 corruptions applied to the MNIST dataset. We choosed the MNIST-C glass blur corruption from 15 corruptions sorts as our second dataset. It has 60,000 images for training, 10,000 for testing, and 10 classes as the MNIST. (c) The notMNIST dataset contains images of the English letters from A to J, thus it consists of 10 classes. The original training set contains 500,000 images and the test part 19,000, from these we selected randomly 60,000 for training and 10,000 for testing. (d) The FashionMNIST dataset has 10 classes as well, and consist on images of Zalando’s articles from T-Shirt, Trouser till Bag, and Ankle boots. It has 60,000 and 10,000 images for training and testing, respectively. (e) The Sign Language MNIST (SLMNIST) dataset contains all the alphabet, except J and Z, of the standard American Sign Language (ASL). It consists of 27,455 and 7,172 images for training and testing, respectively, with 24 classes.

# Experiment design

<!-- With the purpose of selecting the appropriate LSTM architecture and parameter, to classify each image-datasets under the same conditions, first we tested different configurations. The best performance was achieved using the following parameters: three layers, 512 LSTM units per layer, a learning rate of 0.001 with exponential decay, and 256 images per batch size. Also, an early stopping condition has been implemented, consisting on training until there are no changes in
the loss function. The model is compiled with Adam optimizer, the crossentropy loss function and metric accuracy. -->

To identify the optimal LSTM architecture and parameters for image dataset classification under uniform conditions, we conducted preliminary tests with various configurations. The most favorable performance was observed with the following parameters: three layers, each comprising 512 LSTM units, a learning rate of 0.001 with exponential decay, and a batch size of 256 images. Additionally, an early stopping condition was incorporated, wherein training ceases upon observing no further changes in the loss function. The model is compiled using the Adam optimizer, crossentropy loss function, and accuracy as the metric.

<!-- Various training datasets were built for each image dataset, following three different rules. The first rule considers to build the sequences from the rows of the images, which we call horizontal order (H); the second rule consists of building the sequences using the columns of the images, which we call vertical order (V); and the third rule, that we call spiral order (S), consists of building the sequences by collecting the pixels from the center and going out from the center in circular ways till the border of the image. -->

Training datasets were constructed for each image dataset based on three distinct rules. The first rule involves forming sequences from the rows of the images, denoted as horizontal order (H). The second rule entails constructing sequences using the columns of the images, designated as vertical order (V). The third rule, identified as spiral order (S), involves generating sequences by aggregating pixels from the center and expanding outward in circular patterns until reaching the image border.

<!-- A grayscale image in X commonly comes with a M×N pixels, in this work all raw images are 28×28 pixels. From the model’s point of view, varying the shape M × N of an image is equivalent to varying the sequences length and using the sorts of order (H, V and S) is equivalent to varying the order in each sequence. M defines the number of sequences and N defines the length of a sequence that can be got in an image. --> 

A grayscale image typically exhibits dimensions of M×N pixels, with all original images in this study standardized to 28×28 pixels. From the model's perspective, altering the shape M × N of an image is analogous to varying the sequence length, and employing different order types (H, V, and S) is equivalent to diversifying the order within each sequence. Here, M determines the number of sequences, and N establishes the length of a sequence obtainable from an image.

<!-- An experiment E is carried out with a specific combination of M × N shape of the image and a sort order H, V or S as shown by the previous figure. Each experiment yields an accuracy. -->

Experiment E is conducted with a specific combination of M × N image dimensions and one of three sorting orders: H, V, or S, as illustrated in the subsequent figure. Each experiment results in an accuracy measurement.

<img width="250" alt="image" src="https://github.com/ekchacon/Classification-under-Changes-in-Sequences-Order/assets/46211304/25c4099d-8d2a-43f1-b05d-c57468f3c6b8">

The M × N combinations considered in this work are as follow: (2,392), (4,196), (7,112), (8,98), (14,56), (16,49), (28,28), (49,16), (56,14), (98,8), (112,7) and (196,4). These shapes were chosen mainly because they have an integer value
for M and N. As previously mentioned, the datasets used in this work are the well known MNIST, MNIST-C, notMNIST, FashionMNIST and Sign Language MNIST. The combinations of datasets, image shapes and order sorts yield a total of 180 experiments.

# Results and discussion

<!--
Using the best architecture, we evaluated our experiments with the five datasets, organized in different shapes and order of pixels (H, V and S). Our main results are shown in the next table, from which we observe the following:

- Doubtless, the shape (28,28) or sequences with 28 features obtained the maximum accuracies within six experiments, equally distributed between the vertical and horizontal order.
- The shape (2,392) is the second with four maximum accuracies, mostly distributed in the spiral order.
- Generally, the horizontal and vertical order accuracies are slightly above the spiral order accuracies.
- Overall, we observe the maximum accuracies tend to stay above (28,28), where the sequences are longer than those below.
 -->

Utilizing the optimal architecture, we assessed our experiments across five datasets, varying in shapes and pixel orderings (H, V, and S). Key findings are summarized in the subsequent table, revealing the following:

- Undoubtedly, the configuration (28,28) or sequences with 28 features consistently achieved the highest accuracies across six experiments, evenly distributed between vertical and horizontal orderings.
- The configuration (2,392) ranked second with four instances of maximum accuracy, predominantly distributed in the spiral order.
- In general, accuracies for horizontal and vertical orderings slightly exceeded those for spiral orderings.
- Overall, maximum accuracies tended to surpass the baseline of (28,28), indicating superior performance in cases where sequences are longer.

<img width="550" alt="image" src="https://github.com/ekchacon/Classification-under-Changes-in-Sequences-Order/assets/46211304/880412de-5289-4c36-85d6-e5f409b516f5">

<!--
We analyze the accuracies shown in the table  with the help of a boxplot chart, as shown in the next figure. In according to the median and maximum values of the boxplots, we can see the two first datasets (MNIST and MNIST-C) in the H order has better accuracies than the V and S orders. Also, the S order gets the lower accuracies in general. The same pattern appears in the FASHION dataset, though it has a smaller accuracy than MNIST and MNIST-C. 
-->

We conduct an analysis of the accuracies presented in Table with the help of a boxplot chart, as depicted in the following figure. Observing the medians and maximum values from the boxplots, it is evident that the first two datasets (MNIST and MNIST-C) exhibit superior accuracies in the H order compared to the V and S orders. Additionally, the S order consistently yields lower accuracies overall. This pattern is also observed in the FASHION dataset, albeit with a slightly lower accuracy compared to MNIST and MNIST-C.

<!--
The notMNIST and SLMNIST datasets, in the boxplot figure, have different results, the V order has a slightly better median accuracy than the H order median accuracy. Although one dataset is around 95% and the other around 85%, they both follow the same pattern. And the S order has the worst median accuracy in both cases. 
-->

The notMNIST and SLMNIST datasets exhibit divergent outcomes, in the boxplot figure, with the V order presenting a marginally superior median accuracy compared to the H order. Despite distinct overall accuracy levels—approximately 95% for one dataset and 85% for the other—they both conform to a consistent pattern. Notably, the S order consistently yields the lowest median accuracy in both cases.

<!--
On the other hand, we can observe from the boxplot figure, accuracies in the H order are in general less disperse, and the S order has more dispersed accuracies. It means that working with horizontal order is more reliable than the others and it has better median accuracies and are less spread.
-->

On the other hand, as depicted in the boxplot figure, accuracies in the H order exhibit a generally lower dispersion, while the S order displays more dispersed accuracies. This observation implies that working with the horizontal order is characterized by greater reliability, as evidenced by superior median accuracies and reduced spread compared to other orderings.

<img width="550" alt="image" src="https://github.com/ekchacon/Classification-under-Changes-in-Sequences-Order/assets/46211304/dc328fe3-8b49-43b8-baee-5dc8b6a84a59">

# Conclusion

<!--
In this work we presented a procedure to evaluate the accuracy of a recurrent neural network based on LSTM units, for datasets organized in sequences of different order and lengths. The experiments where conducted on five datasets of grayscale images of 28×28 pixels, where changes in sequences order were made by using three different rules and changes in sequences lenghts were made by modifying the shape of the image. The results show that in most of the cases the best accuracies are achieved when the sequences are extracted from images following a horizontal order, but there are some cases where a vertical order can be better.
-->

This study introduces a methodology to assess the accuracy of a recurrent neural network utilizing LSTM units across datasets organized in sequences with varying orders and lengths. Experiments were conducted on five grayscale image datasets, each featuring dimensions of 28×28 pixels. Variations in sequence order were implemented using three distinct rules, and modifications in sequence lengths were achieved by altering the image shape. The findings indicate that, in the majority of cases, optimal accuracies are attained when sequences are extracted in a horizontal order. However, there are instances where a vertical order may yield superior results.
