import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#library for the CNN
#import inspect
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pickle

##############################################################################################
####################### format mangment to make ezpadova properly work########################
##############################################################################################

# Import the encoding to work with ezpadova
if sys.platform == "win32":
    os.environ['PYTHONIOENCODING'] = 'utf-8'
os.system('chcp 65001')

#Load the configuration file to make aviable the configuration variable
try:
    from ezpadova.config import reload_configuration

    reload_configuration()
except:
    pass

#Import the class and the variables to load the isochrones data from CMB website using ezpadova library
from ezpadova.parsec import get_isochrones
from ezpadova.config import configuration

############################################################################################
############################################################################################
################################# Loading Data Section #####################################
############################################################################################
############################################################################################

def load_data(min_age=0.0,max_age=0.0,age_step=0.0,Z_min=0.0152,Z_max=0.03,Z_step=0.0,phot='YBC_gaiaEDR3'):
    #Get the phat to the correct file
    print(configuration.keys())
    photometric_system=configuration['photsys_file'][phot][1]
    
    if min_age==0.0 and max_age==0.0:
        df=get_isochrones(default_ranges=True,photsys_file=photometric_system)
    else:
        df=get_isochrones(age_yr=(min_age,max_age,age_step),Z=(Z_min,Z_max,Z_step),photsys_file=photometric_system)
    
    #Groupping the data as function of logAge and initial metallicity
    groups=df.groupby(['logAge','Zini'])
    
    df_out= dict(list(groups))
    return df_out

######################Simple population functions ######################
#these functions work only for a df that rappresents a simple population#
#with a single value of Zini and logAge.                                #
#########################################################################

def calculate_IMF(df, mini_column='Mini', imf_column='int_IMF'):
    """
    Calculates the IMF (Initial Mass Function) from a given DataFrame.

    Parameters:
    - df: DataFrame containing the 'Mini' and 'int_IMF' columns.
    - mini_column: Name of the column containing the 'Mini' values (default is 'Mini').
    - imf_column: Name of the column containing the 'int_IMF' values (default is 'int_IMF').

    Returns:
    - IMF: Array of interpolated IMF values.
    """
    # Ensure 'Mini' and 'int_IMF' are present in the DataFrame
    if mini_column not in df.columns or imf_column not in df.columns:
        raise ValueError(f"Columns '{mini_column}' and/or '{imf_column}' are not in the DataFrame.")
    
    # Extract the Mini (x-values) and int_IMF (integral) arrays
    Mini = df[mini_column].values  # Array of x
    integral_IMF = df[imf_column].values  # Array of the integral of Y(x)

    # Compute the difference of the integral (Y values)
    d_integral_IMF = np.diff(integral_IMF)  # Successive differences of the integral
    d_Mini = np.diff(Mini)  # Successive differences of Mini (x values)
    
    # Handle division by zero: Replace zeros with a small value (e.g., 1e-10)
    d_Mini[d_Mini == 0] = 1e-10

    # Compute Y(x)
    Y = d_integral_IMF / d_Mini

    # Append the last Y value to ensure the IMF array length matches the original data
    IMF = np.append(Y, Y[-1])

    return IMF

def remove_non_strictly_decreasing(df, imf_array, mini_column='Mini'):
    """
    Removes rows where the IMF array is not strictly decreasing with respect to the 'Mini' column in the DataFrame.

    Parameters:
    - df: The original DataFrame.
    - mini_column: The name of the 'Mini' column.
    - imf_array: The external IMF array that should be strictly decreasing.

    Returns:
    - A DataFrame with rows removed where the IMF array is not strictly decreasing.
    """
    # Sort the DataFrame by the 'Mini' column
    df_sorted = df.sort_values(by=mini_column).reset_index(drop=True)

    # Use the IMF array as-is without sorting (it corresponds directly to the Mini column)
    imf_sorted = imf_array

    # Find rows where the IMF array is not strictly decreasing
    mask_strictly_decreasing = pd.Series(imf_sorted).diff().fillna(0) < 0

    # Apply the mask to the sorted DataFrame
    df_filtered = df_sorted[mask_strictly_decreasing.values]

    return df_filtered

def interpolate_with_equispaced_mini(df, mini_column, num_points):
    """
    Adds equispaced points to the 'Mini' column of a DataFrame and applies interpolation
    to all other columns.

    Parameters:
    - df: the original DataFrame
    - mini_column: the name of the column containing the values to be made equispaced (e.g., "Mini")
    - num_points: the number of equispaced points to generate for the 'Mini' column

    Returns:
    - A new DataFrame with equispaced 'Mini' points and interpolated values.
    """

    # Step 1: Create new equispaced values for the 'Mini' column
    mini_min = df[mini_column].min()
    mini_max = df[mini_column].max()
    new_mini = np.linspace(mini_min, mini_max, num=num_points)

    # Step 2: Create a DataFrame with the new 'Mini' points
    df_new_mini = pd.DataFrame({mini_column: new_mini})

    # Step 3: Merge the two DataFrames on the 'Mini' column with an outer join
    df_merged = pd.merge(df_new_mini, df, on=mini_column, how='outer').sort_values(mini_column)

    # Step 4: Interpolate all numeric columns
    df_interpolated = df_merged.interpolate(method='linear')

    return df_interpolated

def generate_sintetic_diagram_from_isochrone(df, N_samples=10000):
    # Rimuovere duplicati dalla colonna int_IMF
    df_unique = df.drop_duplicates(subset='int_IMF')
    
    
    int_IMF = df_unique['int_IMF']
    
    # Genera numeri casuali uniformi
    uniform = np.random.uniform(np.min(int_IMF), np.max(int_IMF), N_samples)
    
    
    # Trova indici dove int_IMF >= uniform
    indices = np.searchsorted(int_IMF, uniform, side='right')
    
    # Gestisci eventuali indici fuori dal range
    indices = np.clip(indices, 0, len(df_unique) - 1)  # Usa df_unique qui
    
    df_out = df_unique.iloc[indices].copy()  # Copia delle righe selezionate dal DataFrame unico
    return df_out

def generate_cleened_sintetic_diagram_from_isochrone(r,N_samples=1e6):
    
    #Interpolation of the df to better rappresent the IMF with the generated data
    r_interpolate=interpolate_with_equispaced_mini(r,'Mini',int(1e3))
    
    #Evaluation of the IMF for each single value
    IMF_interpolate=calculate_IMF(r_interpolate)
    
    #Elimitation of non decrising IMF to ensure a right generation of the star population
    r_interpolate=remove_non_strictly_decreasing(r_interpolate,IMF_interpolate)
    
    #Re evaluation of the IMF
    IMF_interpolate=calculate_IMF(r_interpolate)
    
    #Generation of the syntetic diagram's values
    df_out_int= generate_sintetic_diagram_from_isochrone(r_interpolate, N_samples=int(N_samples))
    return df_out_int


######################functions for multiple values ####################
#                            of logAge and Zini                        #
########################################################################

def generate_syntetic_diagrams(df_in,N_samples=1e4):
    df_out={}
    for (logage,zini), df in  df_in.items():
        df_out[(logage,zini)]= generate_cleened_sintetic_diagram_from_isochrone(df,N_samples)
        
    return df_out

############################################################################################
############################################################################################
######################## Covolutional Neural Network Section ###############################
############################################################################################
############################################################################################

#########################Immage generation functions########################################

def generate_image_from_histogram(x,y, output_size=(2, 2), cmap='gray', xlim=(-1,5), ylim=(15,-4.5),bins=(200,200),cmin=1):
    """
    Generates an image from a 2D histogram.
    Attentinon: the axis orientation is not formatted as usal in the CMD or HR diagram, but are right handed oriented.
    
    Args:
        x: Array of the x data
        y: Array of the y data
        output_size: Tuple containing the desired dimensions of the output image.
        cmap: Colormap to use for visualization (default: 'gray').
        xlim: Tuple containing the x-axis limits (min, max).
        ylim: Tuple containing the y-axis limits (min, max).
        bins: Tuple of the numbers of the bins (x-bins,y-bins); default: (200,200).
        cmin: The minimum number of entries for bin (default: 1)

    Returns:
        A NumPy array representing the generated image.
    """

    # Create the image
    plt.figure(figsize=output_size)
    plt.hist2d(x,y,bins=bins, cmin=cmin,cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    if xlim!=None:
        plt.gca().set_xlim(xlim)
    if ylim!=None:
        plt.gca().set_ylim(ylim)
        
    # Get the axies limits
    xlim=plt.xlim()
    ylim=plt.ylim()
    axes_lim=xlim+ylim

    # Save the image in memory (as a NumPy array)
    fig = plt.gcf()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))  # RGB image
    plt.close(fig)

    # Convert to grayscale (if needed)
    if cmap == 'gray':
        data = data[:, :, 0]

    return data,axes_lim

def generate_immages_and_labels(df_in,x_key='BP-RP',y_key='Gmag',xlim=(-1,5),ylim=(15,-4.5)):
    immages=[]
    labels=[]
    axies_limits=[]
    for (logage,zini), df in  df_in.items():
        label=(logage,zini)
        if x_key=='BP-RP':
            x=np.array(df['G_BPmag'])-np.array(df['G_RPmag'])
        else:
            x=np.array(df[x_key])
        y=np.array(df[y_key])
        
        #Generate immage
        immage, axies_limit=generate_image_from_histogram(x,y,xlim=xlim,ylim=ylim)
        axies_limits.append(axies_limit)
        immages.append(immage)
        labels.append(label)
    
    return immages,labels,axies_limits

def plot_image(image,ax,axies_lim):
    
    ax.imshow(image, cmap='gray',extent=axies_lim)
    plt.axis('tight')
    plt.tight_layout()

##################################################################################
##################### function to train the CNN ##################################
################################################################################## 

class StarNet_CNN:
    """
    Classe per la creazione e l'addestramento di modelli CNN.

    Fornisce metodi per dividere il dataset in training e validation, calcolare medie e deviazioni standard, creare modelli CNN, e addestrare i modelli.
    """

    def __init__(self):
        """
        Costruttore della classe.
        """
        self.default_file_name='default.pkl'
        self.load_default_model()
        

    def divide_dataset(self, X, y, test_size=0.3, random_state=42):
        """
        Divide il dataset in set di training e validation.

        Args:
            X: Array contenente le features (immagini).
            y: Array contenente i target (label).
            test_size: Proporzione del dataset da utilizzare per la validation.
            random_state: Seed per la riproducibilità.

        Returns:
            X_train, X_val, y_train, y_val: Set di training e validation.
        """
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_val, y_train, y_val


    def create_cnn_model(self):
        """
        Crea un modello CNN di base.

        Args:
            input_shape: Forma dell'input (es. (32, 32, 3)). (Default: viene utilizzato l'input_shape fornito in fase di inizializzazione)

        Returns:
            Modello CNN compilato.
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(2),  # Assumiamo 2 output continui
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
    
    def secure_normalization(self,x, mean, std):
        return (x - mean) / std if np.abs(std) >= 1e-4 else (x - mean)

    def normalize_array(self,data,means=None,stds=None):
        if means is None and stds is None:
            means = np.mean(data, axis=0)
            stds = np.std(data, axis=0)
        return np.array([self.secure_normalization(data[:, i], means[i], stds[i]) for i in range(data.shape[1])]).T , (means,stds)
    
    def train_model(self, X, y,X_val=None,y_val=None, model='default', epochs=30, batch_size=10,test_size=0.2, random_state=42):
        """
        Addestra il modello CNN.

        Args:
            model: Modello CNN da addestrare.
            X_train, y_train: Set di training.
            X_val, y_val: Set di validation.
            epochs: Numero di epoche.
            batch_size: Dimensione del batch.

        Returns:
            history: Oggetto che contiene la cronologia dell'addestramento.
        """
        if X_val is None and y_val is None:
            # Esempio d'uso:
            X =  np.array(X)
            y =  np.array(y)

            # Convert X to NumPy array and add channel dimension if necessary
            if not isinstance(X, np.ndarray):
                X = np.array(X)
            if X.ndim == 3:
                X = np.expand_dims(X, axis=-1)

            # Dividi il dataset
            X_train, X_val, y_train, y_val = self.divide_dataset(X, y,test_size, random_state)
        else:
            X_train=X
            X_val=X_val
            y_train=y
            y_val=y_val
        
        self.input_shape=X_train.shape[1:]
        
        if model=='default':
            model=self.create_cnn_model()
        
        #Linearization of the logAge range
        y_val[:,0]=np.power(10,y_val[:,0])
        y_train[:,0]=np.power(10,y_train[:,0])
        
        #Normalization
        y_train,self.mean_std=self.normalize_array(y_train)
        y_val,_=self.normalize_array(y_val,self.mean_std[0],self.mean_std[1])
        
        self.history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
        self.model=model
        
    def plot_loss(self):
        # Plot della loss
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper right')
        plt.gca().set_yscale('log')

    def plot_mae(self):
        # Plot del MAE
        plt.plot(self.history.history['mae'])
        plt.plot(self.history.history['val_mae'])
        plt.title('Model MAE')
        plt.ylabel('MAE')
        plt.xlabel('Epoch')
        plt.legend(['Train',  'Val'], loc='upper right')
        plt.gca().set_yscale('log')

    def secure_denormalization(self,x, mean, std):
        """
        Denormalizza i dati normalizzati utilizzando media e deviazione standard.

        Args:
            x: Array di dati normalizzati.
            mean: Media originale utilizzata per la normalizzazione.
            std: Deviazione standard originale utilizzata per la normalizzazione.

        Returns:
            Dati denormalizzati.
        """
        return (x * std) + mean if np.abs(std) >= 1e-4 else (x + mean)

    def denormalize_array(self,normalized_data, means, stds):
        """
        Denormalizza un array di dati utilizzando le medie e deviazioni standard.

        Args:
            normalized_data: Array di dati normalizzati.
            means: Media originale per ciascuna colonna.
            stds: Deviazione standard originale per ciascuna colonna.

        Returns:
            Dati denormalizzati.
        """
        return np.array([self.secure_denormalization(normalized_data[:, i], means[i], stds[i]) for i in range(normalized_data.shape[1])]).T
    
    def predict(self,x):
        prediction=self.model.predict(x)
        return self.denormalize_array(prediction,self.mean_std[0],self.mean_std[1])
    
    
    ###########################################################################
    #######################Load and write models###############################
    ###########################################################################
    
    def save_to_file(self, filename):
        """
        Salva l'istanza corrente della classe in un file usando pickle.

        Args:
            filename (str): Il percorso e il nome del file in cui salvare l'istanza.
        """
        try:
            with open(filename, 'wb') as f:
                pickle.dump(self, f)
            print(f"Classe salvata correttamente in {filename}")
        except Exception as e:
            print(f"Errore nel salvataggio: {e}")
            
    def load_from_file(self, file_path):
        """
        Carica lo stato dell'istanza da un file pickle.

        Args:
            file_path: Il percorso del file pickle da caricare.
        """
        try:
            
            with open(file_path, 'rb') as f:

                loaded_instance = pickle.load(f)
            
            # Sovrascrivi i dati dell'istanza corrente con quelli caricati
            self.model = loaded_instance.model
            self.mean_std = loaded_instance.mean_std
            self.input_shape=loaded_instance.input_shape
            self.history=loaded_instance.history
            # Puoi aggiungere ulteriori attributi da sovrascrivere se necessario

            print("Modello e dati caricati con successo.")
        except Exception as e:
            print(f"Errore nel caricamento: {e}")

    def get_default_models_folder_path(self):
        # Ottieni la directory corrente (dove si trova il file attuale)
        current_dir = os.path.dirname(__file__)

        # Risali alla cartella principale di StarNet (se sei in una sotto-libreria)
        starnet_dir = os.path.abspath(current_dir)
        
        #starnet_path = inspect.getfile(StarNet)
        #starnet_dir = os.path.dirname(starnet_path)

        #parte comune
        default_models_dir=os.path.join(starnet_dir,'default_models')
        return default_models_dir
    
    def save_default_model(self):
        default_models_dir=self.get_default_models_folder_path()
        
        default_model_file_path=os.path.join(default_models_dir,self.default_file_name)
        
        self.save_to_file(default_model_file_path)
    
    def load_default_model(self):
        default_models_dir=self.get_default_models_folder_path()
        
        default_model_file_path=os.path.join(default_models_dir,self.default_file_name)
        
        self.load_from_file(default_model_file_path)
             
print('training classes imported')
