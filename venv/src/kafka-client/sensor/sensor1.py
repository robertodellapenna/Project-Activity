from kafka import KafkaProducer
import pickle
import pandas as pd
import numpy


def main():
    # leggo il csv
    scania_train_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00421/aps_failure_training_set.csv'
    scania_train = pd.read_csv(scania_train_url, header=14, na_values='na')

    # sostituisco i missing values con la media della colonna
    scania_train.fillna(scania_train['ab_000'].mean(), inplace=True)

    # utilizzo le ultime 58k istanze del dataset come se fossero nuovi dati da classificare
    new_istances = scania_train.drop('class', axis=1).tail(29000)

    # converto tutti i float in interi
    new_istances = new_istances.astype('int64')

    producer = KafkaProducer(bootstrap_servers=['localhost:9093'],
                             value_serializer=lambda x: pickle.dumps(x))

    for istance in new_istances.iterrows():
        producer.send('sensor-fog', istance, partition=1)
        producer.flush()


if __name__ == "__main__":
    main()
