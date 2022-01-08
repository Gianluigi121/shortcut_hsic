# shortcut_hsic
main_aux.py: Code for removing shortcut learning by imposing the hsic standard

age_model.py: Predict age based on the chest x-ray

Fashion_MNIST_age.ipynb:
1. Preprocess the dataset from Kaggle and select 2000 samples for this task
2. Assign a continuous "age" value to each sample:
    - mean = (label+1)*5, range from 5-50
    - Age is sampled from the gaussian distribution N~(mean, 1)
    - Age approximately range from 3-52
3. For each sample, apply a gaussian filter with the scale(variance) determined by the age value
    - scale = age/10 (float number)
    - scale approximately range from 0.3-5.2 
4. Train a VGG16 model to predict the age based on the processed image
    - epochs = 10, mean squared error
