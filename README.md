# COMP411: Computer Vision with Deep Learning
**College of Engineering**

**COMP 411 – Computer Vision with Deep Learning Final Project**

**PokéGAN - Pokémon Type Casting for Generation One Pokémon using Generative Adversarial Networks**

**Fall 2022**

**Participant Information:**
- Oya Suran - oyasuran18@ku.edu.tr

- Ata Sayın - asayin18@ku.edu.tr

- Kerem Girenes - kgirenes18@ku.edu.tr

## Table of Contents
- [Introduction](#introduction)
- [Motivation](#motivation)
- [Dataset](#dataset)
- [Model Choice](#model-choice)
- [Limitations and Assumptions](#limitations-and-assumptions)
- [Model Results and Development Stages](#model-results-and-development-stages)
- [Network Architecture](#network-architecture)
- [Training Details](#training-details)
- [Results](#results)
- [Fire2grass Results](#fire2grass-results)
- [Fire2water Results](#fire2water-results)
- [Fire2electric Results](#fire2electric-results)
- [Model Comparisons](#model-comparisons)
- [Cycle Consistency](#cycle-consistency)
- [Black Artifacts and Poor Transformation Examples in All of our Models](#black-artifacts-and-poor-transformation-examples-in-all-of-our-models)
- [Discussion](#discussion)
- [Fluctuating Results with Growing Epochs](#fluctuating-results-with-growing-epochs)
- [Playing around with the learning rate to solve the black artifacts problem](#playing-around-with-the-learning-rate-to-solve-the-black-artifacts-problem)
- [Using 128 x 128 image models for shorter training time](#using-128-x-128-image-models-for-shorter-training-time)
- [Using the original batch size to overcome black artifacts](#using-the-original-batch-size-to-overcome-black-artifacts)
- [Pokémon types having too similar colors and textures](#pokemon-types-having-too-similar-colors-and-textures)
- [Mode collapse](#mode-collapse)
- [Expanding the dataset](#expanding-the-dataset)
- [Loss](#loss)
- [Future Work](#future-work)
- [Conclusion](#conclusion)
- [References](#references)
- [Appendix](#appendix)

## Introduction
Our project revolves around creatures called Pokémon in Pokémon anime, films, and video games. Each Pokémon has unique power and skill sets with different personalities and designs. Every Pokémon belongs to a type category such as water, ground, fairy, etc. Furthermore, Pokémon can be categorized by generations as well. Generation of Pokémon represents which period a specific Pokémon made its debut, for example, the first generation would correspond to first seeing this Pokémon in the years between (1996–1999).

In our project, we focused on the first generation Pokémons. We used Generative Adversarial Networks (GANs), more specifically CycleGANs, to transform the appearance of a Pokémon from one type to another. For example, Charmander is a first-generation fire type Pokémon. When we provide a Charmander image and use our fire2electric model, our model converts Charmander into an electric type version of it. To make this type conversion possible, we researched different GANs and decided we should base our CycleGAN model on the paper “Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks'' by Zhu et al. Our main objective was to effectively convert a Pokémon's appearance to match a different type while maintaining the plausibility of the original image. We made different design choices which are going to be explained in this report.

## Motivation
Initial project idea was to construct and train a model that would generate a new, never-seen-before Pokémon. However, as we were researching applicable datasets, we saw that there were some projects and groups both on Github and Kaggle who had already implemented this idea. Some of their outputs were great, some of them were not successful since the Pokémon generated were blurry and did not actually seem like a genuine Pokémon. Thus, failed cases gave us an idea to use existing Pokémon and its base form, which could tackle the blurring issue and ingenuity. Afterwards, we talked with our TAs and concluded that our project idea was going to be type changing for Generation 1 Pokémon. Generation 1 was chosen because most ‘authentic’ Pokémon are in Generation 1, and there were more complete datasets for Generation 1 than the other generations. We decided to use CycleGAN since we do not have paired data. Furthermore, to the best of our knowledge, CycleGAN has not ever been used in our proposed manner, and we believe converting Pokémon to different types is both a fun and an educational project idea. Additionally, our project can fill the gap in visual representation of all possible Pokémon type combinations, and it has the potential for creative use in fan art and games.

## Dataset
At first, we started with a small dataset compared to the one we currently have. The image datasets did not include the type of the Pokémon, so we used another dataset that includes the types and stats of the Pokémon’s. We matched these types with the corresponding names in the resulting image dataset, and used this finalized dataset as our input to our models. To preprocess and link images with their types we used Jupyter notebook. You can see our code in our GitHub repository under the dataset directory “PokeGAN Dataset Preprocessing.ipynb”. In this notebook, we first searched for names of the first-generation Pokémon since our dataset found in Kaggle included other generations of Pokémon as well. After that, we categorized Pokémon according to their type (‘fire’, ‘grass’, etc.). Then, we moved each Pokémon to a folder named their type. Moreover, we made a decision to start with fire, electric, and grass type Pokémon. Therefore, we divided our dataset to train and test for each type. These sets are located in the dataset folder. Each image had different sizes which is not the ideal case to train them. Thus, using the Pillow library, we converted each image to be 256 x 256. As we mentioned in our progress reports, some of the previous results in ‘electric’ to ‘fire’ transformation were not ideal, and increasing the dataset could have solved this problem. Thus, we looked into different datasets and found one with 2GB of Generation 1 Pokémon images. We downloaded the dataset and preprocessed it with our existing preprocessing code. Our initial dataset consists of a total of 6,705 images. Our new complete dataset has more than 30,000 images. However one must note, we only used images in ‘Grass’, ‘Electric’,’Fire’ and ‘Water’ types directories. Previously, each type of Pokémon had around 400 training images. After expanding the dataset, Fire Type had around 1400 images and Water Type had around 4400 in the training dataset.

**Image Datasets:**
- Initial Dataset (6,705 images) [Dataset Link](https://www.kaggle.com/saikrithik/pokemon)
- Expanded Dataset (30,000+ images) [Dataset Link](https://www.kaggle.com/vishalsubbiah/pokemon-images-and-types)
- Image Resolution: 256 x 256 pixels

## Model Choice
For this project, we decided to use CycleGAN as our primary model. CycleGAN is well-suited for unpaired image-to-image translation tasks, which is the case in our project. We don't have paired data for Pokémon type transitions, and CycleGANs allow us to translate images from one type to another without the need for explicit pairs. This flexibility and the ability to handle a wide range of image translation tasks make CycleGANs an excellent choice for this project.

## Limitations and Assumptions
Throughout the project, we faced several limitations and challenges. Here are some of the main limitations and the assumptions we made to address them:

**Black Artifacts:** One of the challenges we encountered was the presence of black artifacts in some of the generated images. To address this, we made several assumptions about adjusting the batch size and learning rate during training. These assumptions were tested in the hope of reducing or eliminating these artifacts.

**Dataset Distribution:** The distribution of Pokémon types in our dataset was not uniform, leading to imbalances in our models. This affected the results for some types and led to mode collapse. We assumed that expanding the dataset for certain types might help balance this issue.

**Mode Collapse:** In some cases, our models exhibited mode collapse, where they generated similar images regardless of the input. This could be due to the imbalanced dataset and the similarity of textures and colors in some Pokémon types.

## Model Results and Development Stages
Our project involved training three models: fire2grass, fire2water, and fire2electric. Each model was trained for varying numbers of epochs and with specific datasets. Here's a summary of the key development stages, network architectures, and training details:

## Network Architecture
CycleGAN consists of two generators and two discriminators. Each generator learns to translate images from one domain to another, while each discriminator aims to distinguish between real and generated images. The generators consist of an encoder-decoder architecture, and the discriminators are based on convolutional neural networks (CNNs).

## Training Details
The training process of our models involved multiple iterations and adjustments. We experimented with various hyperparameters, such as learning rates, batch sizes, and the number of epochs. Here are some of the key training details:

## Results
Our project aimed to create convincing transformations from one Pokémon type to another. Here are some of the results we obtained for each type transformation model:

### Fire2grass Results
In our initial fire2grass transformation model, the results varied depending on the number of training epochs. In some cases, the model successfully transformed fire-type Pokémon into grass-type, producing plausible images. However, there were also instances where the results contained noticeable artifacts and deviations from the target type. Here are some sample results:

<img width="1104" alt="Ekran Resmi 2023-11-01 09 57 33" src="https://github.com/keremgirenes/COMP411-PokeGAN/assets/69321438/91b09840-7a6f-4356-b551-64126b510c5e">

Charmander (Fire type Pokémon) converted to Grass type with varying epochs.

<img width="1145" alt="Ekran Resmi 2023-11-01 09 58 59" src="https://github.com/keremgirenes/COMP411-PokeGAN/assets/69321438/82a2af50-ca9d-4f2a-a41a-f08f265705ab">

Bulbasaur (Grass type Pokémon) converted to Fire type with varying epochs.

### Fire2water Results
Similar to the fire2grass transformation, the fire2water transformation model exhibited varying degrees of success depending on the training epoch. Some results achieved a convincing transformation from fire to water type, while others contained artifacts and deviations. Here are sample results:

<img width="1107" alt="Ekran Resmi 2023-11-01 10 00 40" src="https://github.com/keremgirenes/COMP411-PokeGAN/assets/69321438/ad7a6d61-f520-439c-980a-c4b54c3f5d19">

Test outputs of fire2water model over varying training epochs.

### Fire2electric Results
Our fire2electric transformation model showed mixed results as well. Depending on the training epoch, the model generated plausible electric-type transformations from fire-type Pokémon. However, there were also instances of artifacts and less convincing transformations. Here are sample results:

<img width="753" alt="Ekran Resmi 2023-11-01 10 01 32" src="https://github.com/keremgirenes/COMP411-PokeGAN/assets/69321438/7a82e9ec-8547-4c56-8b42-0182404d6276">

Test outputs of fire2electric model over varying training epochs.

## Model Comparisons
In comparing the results of our three transformation models, we observed differences in the quality of the transformations and the presence of artifacts. The fire2grass transformation model generally produced better results than the fire2water and fire2electric models.

<img width="753" alt="Ekran Resmi 2023-11-01 10 01 59" src="https://github.com/keremgirenes/COMP411-PokeGAN/assets/69321438/c9091bfb-144c-4c49-8061-53e3e0795791">

Rapidash (Fire type Pokémon) converted to Grass type, Water type, and Electric type after each model’s training is finalized.

## Cycle Consistency
One of the key principles of CycleGAN is cycle consistency. This means that if we convert an image from one domain to another and then back to the original domain, it should closely resemble the original image. Ensuring cycle consistency is essential for producing high-quality transformations.

<img width="753" alt="Ekran Resmi 2023-11-01 10 02 25" src="https://github.com/keremgirenes/COMP411-PokeGAN/assets/69321438/76611e7e-bf5c-4141-ba89-dc23b0a5d178">

Charmander (Fire type Pokémon) converted to Grass type, then back to Fire type with fire2grass model.

## Black Artifacts and Poor Transformation Examples in All of our Models
We encountered challenges related to black artifacts and poor transformations in our models. These issues were particularly prominent in the fire2electric transformation, and we made various attempts to address them.

<img width="764" alt="Ekran Resmi 2023-11-01 10 02 49" src="https://github.com/keremgirenes/COMP411-PokeGAN/assets/69321438/792bb66c-8f2b-4617-8600-02c9e110c4da">

Test outputs showing the black artifacts and poor transformations

## Discussion
In the course of this project, we had various discussions and observations related to the model's performance and the challenges we faced. Here are some key points of discussion:

### Fluctuating Results with Growing Epochs
One observation we made was the fluctuation in results as the number of training epochs increased. Some transformations improved with additional training, while others became worse or exhibited artifacts. This behavior was somewhat unpredictable and led to variations in the quality of the generated images.

### Playing around with the learning rate to solve the black artifacts problem
The presence of black artifacts in some images was a persistent issue. We experimented with different learning rates in the hope of reducing these artifacts, but the results were not consistently improved.

<img width="590" alt="Ekran Resmi 2023-11-01 10 03 52" src="https://github.com/keremgirenes/COMP411-PokeGAN/assets/69321438/615438aa-f185-4ebe-bf65-e0db678743eb">

Charmander to Grass type with similar amount of training and different learning rates.

### Using 128 x 128 image models for shorter training time
To address the long training time required for 256 x 256 image models, we explored the use of lower resolution images (128 x 128 pixels). While this significantly reduced training time, it also impacted the quality of the transformations, resulting in less detailed images.

### Using the original batch size to overcome black artifacts
To combat black artifacts, we decided to use the original batch size recommended in the CycleGAN paper (batch size of 1). While this reduced the occurrence of black artifacts, it also led to slower training and required additional computational resources.

### Pokémon types having too similar colors and textures
In some cases, we observed that certain Pokémon types had very similar colors and textures, making it challenging for the model to perform accurate transformations. This similarity often resulted in images that were less distinct from the original type.

### Mode collapse
Mode collapse was another challenge we faced, where the model generated similar images regardless of the input. This was particularly evident when training with imbalanced datasets.

### Expanding the dataset
To address the issue of imbalanced datasets and improve the results, we decided to expand the dataset. This expansion aimed to include more examples of certain Pokémon types to ensure a more balanced distribution.

<img width="516" alt="Ekran Resmi 2023-11-01 10 04 45" src="https://github.com/keremgirenes/COMP411-PokeGAN/assets/69321438/7d02a3f8-d197-43be-ac03-24d3151b5ec0">

### Loss
We also discussed the importance of monitoring and interpreting the generator and discriminator losses during training to gain insights into the model's performance and stability.

<img width="803" alt="Ekran Resmi 2023-11-01 10 05 37" src="https://github.com/keremgirenes/COMP411-PokeGAN/assets/69321438/800f012a-da59-4546-a129-12e0da62135a">

Fire2electric generator generator losses

## Future Work
While we have made significant progress in transforming Pokémon types using CycleGANs, there is room for further improvement. Here are some suggestions for future work:

## Conclusion
In conclusion, our project aimed to convert the types of Generation One Pokémon using Generative Adversarial Networks, primarily CycleGANs. We achieved some successful transformations, demonstrating the potential of this approach. However, several challenges, such as black artifacts, mode collapse, and dataset imbalances, remain to be addressed. Further tuning and the availability of additional computational resources would be necessary to enhance the results. Despite the limitations, our project offers a creative and fun way to explore the world of Pokémon through image-to-image translation.

## References
Our project was inspired by various sources, papers, and previous GAN implementations. Here are some of the references we found useful:

[1] The Official Pokémon Website (November 21, 2022) Pokémon.com. Available at: https://www.Pokémon.com/us/ (Accessed: November 21, 2022).

[2] J.-Y. Zhu, T. Park, P. Isola, A. A. Efros, “Unpaired image-to-image translation using cycle-consistent adversarial networks”, IEEE International Conference on Computer Vision.

[3] Philadelphia Museum of Art [@pokem0n.in]. (2022, December 27).Pokem0n.in’s Post”[Photograph]. Instagram. https://www.instagram.com/p/CmrAZCgLUHn/?igshid=Zjc2ZTc4Nzk=

[4] Zhang, Han and Goodfellow, Ian and Metaxas, Dimitris and Odena, Augustus, “Self-Attention Generative Adversarial Networks”

[5] Google. (n.d.). Common problems | machine learning | google developers. Google. Retrieved January 23, 2023, from https://developers.google.com/machine-learning/gan/problems

[6] Jin, Xiaohan & Qi, Ye & Wu, Shangxuan. (2017). CycleGAN Face-off

[7] Luo, L., Hsu, W., & Wang, S. (2021, January). Shape-aware generative adversarial networks for attribute transfer. In Thirteenth International Conference on Machine Vision (Vol. 11605, pp. 328-334). SPIE.

[8] Chan, E. R., Lin, C. Z., Chan, M. A., Nagano, K., Pan, B., De Mello, S., ... & Wetzstein, G. (2022). Efficient geometry-aware 3D generative adversarial networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 16123-16133).

[9] Tang, H., Liu, H., Xu, D., Torr, P. H., & Sebe, N. (2021). Attentiongan: Unpaired image-to-image translation using attention-guided generative adversarial networks. IEEE Transactions on Neural Networks and Learning Systems.

## Appendix (Source Code)
For detailed information about our project's source code and additional materials, please refer to our GitHub repository.
