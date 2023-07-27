import pickle, time
from data_loader import load_data
from data_processing import data_processing
from signal_generator import generate_signal
from configs import MODELS, MODEL_FILE_PATH, SHORT_PERIOD, TRADE_RANGE, SLEEP_TIME
import datetime

def main():
    # load models
    models_dict = {model: None for model in MODELS}
    for model in MODELS:
        with open(MODEL_FILE_PATH.format(name=model), "rb") as model_file:
            models_dict[model] = pickle.load(model_file)
            model_file.close()

    for i in range(TRADE_RANGE):
        # load last data
        stock_data = load_data(period=SHORT_PERIOD)
        processed_data = data_processing(stock_data=stock_data)

        now = datetime.datetime.now()
        # generate signal
        for model in MODELS:
            generate_signal(
                model=models_dict[model], model_name=model, stock_data=processed_data, time=now
            )

        time.sleep(SLEEP_TIME)


main()
