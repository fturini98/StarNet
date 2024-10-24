# StarNet
Library for identifing the properties of a star's simple population using a convolutional Neural Network

## Example Usage
The library works on four step:

1) Loading the isochrones from the [CMD website](http://stev.oapd.inaf.it/cgi-bin/cmd) using [ezpadova](https://github.com/mfouesneau/ezpadova) package:
    ```
    #Import the train module
    import StarNet.train

    #Load the isochrones from the CMD website
    isochrones=StraNet.train.load_data(min_age=5e8,max_age=1e10,age_step=1e8,Z_step=0.005)
    ```
    - The loaded isochrones are presented as a pandas DataFrame where the keys are each one a tuple containig the log(Age) and the initial maetallicity (Z) of each population retrived form the CMD site.

2) Transform the isochrones in synthetic diagrams:
    ```
    #Generate the DataFrame containing the synthetic diagrams
    synthetics=StraNet.train.generate_synthetic_diagram_from_isochrone(df,Nsamples=1e4)
    
    #Generates the arraies to train the CNN
    immages,labels,axies_limits=StraNet.train.generate_immages_and_labels(synthetics)
    ```

    - The default settings for generating the immages are aviable in the [source code](StarNet/train.py), the most relevant are:

        - xlim=(-1,5)
        - ylim=(15,-45)
        - x_key='BP-RP'
        - y_key='Gmag'

        Therfore the standar immages are generated to train the CNN for the GAIA photometry.

    - Is possibple to show the immages as follow:
        ```
        import matplotlib.pyplot as plt
        _,ax=plt.subplots(1,1,figsize=(12,12))
        StraNet.train.plot_image(immages[0],ax,axies_limits[0])
        ```

3) Create the CNN model and train it:
    ```
    import StarNet

    #Import the CNN class ('the class constructor try to 
    #import the default.pkl model if it's present in the 
    #default_models folder, else raise an error, but the 
    #class is redy to be trained')
    CNN=StraNet.CNN()

    #Train the model
    CNN.train_model(immages,labels,X_val=None,y_val=None, model='default', epochs=30, batch_size=10,test_size=0.2, random_state=42)
    ```
    - The standard CNN model has(other model must be implemented):
        - 2 Convolutional layer followed by a Max pooling layer
        - 1 Flatten layer
        - 1 Dense layer coposed by 128 neuron
        - 1 Dense layer coposed by 2 neuron to estimate the 2 value
    - Is possible to pass a validation set, in this case the array immages and labels are used only for the training
    - The labels for the different immages must be compose by a tuple containing (log(age),Z)

4) Predict the age and the initial metallicty of a simple population:
    ```
    #Assuming a DataFrame containg the data of an CMD of a simple population
    immage_to_predict=StarNet.generate_immage(df,x_key,y_key)

    #Predict the age and metallicity the population
    prediction=CNN.predic(immage_to_predict)
    age=prediction[0][0] #age in yr
    Z=prediction[0][1]#Metallicity
    ```
    - NOTE: the prediction is given as an array containing a 2X1 array where the first number rappresent the age in yr and the second the metallicity

5) It's also possible to save and load the trained CNN from the [default_models](StarNet/default_models) folder:
    ```
    #Choose the name of the model
    CNN.default_file_name='my_default.pkl'

    #Save model
    CNN.save_default_model()#Save the file with the name CNN.default_file_name

    #load model
    CNN.load_default_model()#load the file with the name CNN.default_file_name

    ```
## Usefull fucntionalities

### CNN class
The StarNet.CNN class as also other usefull functionality:
```
#Show the graph about the loss function obtained during the training
CNN.plot_loss()

#Show the graph about the Mean Absolute Error function obtained during the training
CNN.plot_mae()
```