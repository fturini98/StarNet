from .train import StarNet_CNN
from .train import generate_image_from_histogram
import numpy as np

class CNN(StarNet_CNN):
    def __init__(self):
        super().__init__()  # Chiama il costruttore della classe padre
        
def generate_immage(df,x_key='BP-RP',y_key='Gmag',xlim=(-1,5),ylim=(15,-45)):
    if x_key=='BP-RP':
        x=np.array(df['G_BPmag'])-np.array(df['G_RPmag'])
    else:
        x=np.array(df[x_key])
    y=np.array(df[y_key])
        
    #Generate immage
    immage, _=generate_image_from_histogram(x,y,xlim=xlim,ylim=ylim)
    single_immage= np.expand_dims(immage, axis=0)
    return single_immage