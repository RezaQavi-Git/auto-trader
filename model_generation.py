from data_loader import load_data
from data_processing import data_processing
from prediction_models import prediction_models 
from signal_generator import generate_signal
from configs import LONG_PERIOD, SHORT_PERIOD, MODEL_FILE_PATH
import pickle

def dump_model(model, name):
    # save the model
    with open(MODEL_FILE_PATH.format(name=name), 'wb') as file:
        pickle.dump(model, file=file)
        file.close()
  
def model_generator():
    # load long term data 
    stock_historical_data = load_data(period=LONG_PERIOD)
    # data processing 
    processed_data = data_processing(stock_data=stock_historical_data)
    # prediction models
    rf_model, gb_model, svm_model = prediction_models(processed_stock_data=processed_data)

    dump_model(rf_model, 'RF')
    dump_model(gb_model, 'GB')
    dump_model(svm_model, 'SVM')
    return


model_generator()

