import numpy as np
from keras.models import load_model
from matplotlib.image as processimage

model = load_model('model')

class PredictImage:
    def __init__(self):
        pass
    
    def predict(self, filename):
        pred_img = processimage.imread(filename)
        pred_img = np.array(pred_img)
        pred_img = pred_img.reshape(-1, 28, 28, 1)
        prediction = model.predict(pred_img)
        final_prediction = [result.argmax() for result in prediction][0]
        return final_prediction

if __name__ = '__main__':
    filename = '' # fill in image path you want to predict
    Predict = PredictImage()
    result = Predict.predict(filename)
    print('classify result is:' , result)