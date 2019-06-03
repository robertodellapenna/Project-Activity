from threading import Thread, Semaphore
import numpy as np
import time
import pandas as pd
import pickle
import sys
import datetime
from kafka import KafkaConsumer, KafkaProducer, TopicPartition
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score

sp = Semaphore()
sts = Semaphore()
num_predictions = 0
num_false_negatives = 0
model = DecisionTreeClassifier()
supervised = pd.DataFrame()
train_set = pd.DataFrame()
predictions = pd.DataFrame()
columns = []
fixed = 2000
model_version = 1


class BColors:
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


def update_model_remote():
    global model, train_set, fixed, model_version

    consumer = KafkaConsumer('cloud-fog',
                             group_id='model_update1',
                             bootstrap_servers=['localhost:9093'])

    # leggo il modello ed i dati di train dal Cloud
    print("In attesa del modello dal Cloud..")
    for message in consumer:
        offset = message.offset
        key = message.key.decode("utf-8")
        if offset > 1:
            if key == "model":
                # deserializzo e estraggo il valore
                tmp_model = pickle.loads(message.value)
                size = sys.getsizeof(tmp_model)
                t = datetime.datetime.now()
                print(BColors.OKGREEN + str(t) + " - Modello ricevuto, versione " + str(model_version) + ", peso " + str(size) + "bytes" + BColors.ENDC)
                model_version = model_version + 1

                message = next(consumer)
                key = message.key.decode("utf-8")
                if key == "train":
                    # differenza tra train set iniziale e attuale
                    diff_train = (np.frombuffer(message.value, dtype='int64'))

                    num_cols = int(len(columns) + 1)
                    num_rows = int(len(diff_train)/num_cols)

                    diff_train = pd.DataFrame(diff_train.reshape([num_rows, num_cols]), columns=np.append(['class'], columns))

                    # aggiorno modello e train set
                    model = tmp_model

                    sts.acquire()
                    first_train = train_set.head(fixed)
                    train_set = first_train.append(diff_train)
                    sts.release()

                    fixed = fixed + num_rows

                else:
                    print(BColors.FAIL + "Train assente" + BColors.ENDC)
            else:
                print(BColors.FAIL + "Modello assente" + BColors.ENDC)


def consume_sensor_data():
    global sp, model, predictions, columns

    consumer = KafkaConsumer(bootstrap_servers=['localhost:9093'],
                             value_deserializer=lambda x: pickle.loads(x))
    consumer.assign([TopicPartition('sensor-fog', 1)])

    print("In attesa dei dati dai sensori..")
    # leggo i dati dai sensori
    for message in consumer:
        istance = message.value[1]
        prediction = pd.DataFrame([istance])
        prediction = prediction[columns]
        predicted = model.predict(prediction)[0]
        prediction.insert(0, 'class', predicted)

        sp.acquire()
        predictions = predictions.append(prediction)
        sp.release()


def update_model_local():
    global sp, sts, num_predictions, supervised, train_set, predictions, num_false_negatives

    while True:
        time.sleep(120)
        print("Aggiornamento locale")

        if len(predictions) == 0:
            print("Nessuna nuova predizione disponibile")

        else:
            false_negatives = []

            sp.acquire()
            for n in range(len(predictions.index)):
                predicted = predictions['class'].iloc[n]
                index = num_predictions + n
                true = supervised['class'].iloc[index]
                # filtro solo i falsi negativi
                if predicted == 0 and true == 1:
                    false_negatives.append(index)
            nfn = len(false_negatives)
            npd = len(predictions.index)
            print(BColors.OKBLUE + str(nfn) + " falsi negativi trovati in " + str(npd) + " predizioni" + BColors.ENDC)
            num_false_negatives = num_false_negatives + len(false_negatives)

            num_predictions = num_predictions + len(predictions.index)
            tfn = num_false_negatives
            tpd = num_predictions
            print(BColors.OKBLUE + str(tfn) + " falsi negativi totali su " + str(tpd) + " predizioni totali" + BColors.ENDC)

            # aggiorno train set
            if len(false_negatives) > 0:

                sts.acquire()
                train_set = train_set.append(supervised.take(false_negatives))

                # creo il modello
                features_train_new = train_set.drop('class', axis=1)
                labels_train_new = train_set['class']

                new_model = DecisionTreeClassifier()
                new_model.fit(features_train_new, labels_train_new)

                producer = KafkaProducer(bootstrap_servers=['localhost:9093'])

                # invio il modello
                serialized = pickle.dumps(new_model)
                size = sys.getsizeof(new_model)
                producer.send('fog-cloud', key=b"model", value=serialized, partition=1)

                t = datetime.datetime.now()
                print(BColors.OKBLUE + str(t) + " - Modello inviato al cloud, peso: " + str(size) + " bytes" + BColors.ENDC)

                diff_train = train_set.iloc[fixed:len(train_set.index), :]
                serialized = diff_train.to_numpy().tobytes()
                producer.send('fog-cloud', key=b"train", value=serialized, partition=1)

                print("%d istanze di train inviate" % len(diff_train.index))

                producer.flush()
                sts.release()

            predictions = pd.DataFrame()
            sp.release()


def setup():
    global train_set, supervised

    # leggo il csv
    scania_train_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00421/aps_failure_training_set.csv'
    scania_train = pd.read_csv(scania_train_url, header=14, na_values='na')

    # sostituisco i missing values con la media della colonna
    scania_train.fillna(scania_train['ab_000'].mean(), inplace=True)

    # filtro gli attributi corrispondenti agli indici ricevuti dal cloud
    scania_train = scania_train[np.append(['class'], columns)]

    # sostituisco i valori della classe con interi
    scania_train['class'].replace(to_replace=dict(pos=1, neg=0), inplace=True)

    # converto tutti i float in interi
    scania_train = scania_train.astype('int64')

    # salvo le ultime 58k istanze come dati supervisionati
    supervised = scania_train.tail(29000)

    # salvo le prime 2k istanze utilizzate per costruire il modello iniziale
    train_set = scania_train.head(2000)


def main():
    global model, columns, model_version
    # leggo il modello iniziale e il subset di attributi dal nodo cloud
    consumer = KafkaConsumer('cloud-fog',
                             group_id='model_update1',
                             bootstrap_servers=['localhost:9093'],
                             value_deserializer=lambda x: pickle.loads(x))

    print("In attesa del modello iniziale dal Cloud..")
    message = next(consumer)
    key = message.key.decode("utf-8")
    if key == 'model':
        model = message.value
        print(BColors.OKGREEN + "Modello ricevuto, versione " + str(model_version) + BColors.ENDC)
        model_version = model_version + 1

        message = next(consumer)
        key = message.key.decode("utf-8")
        if key == 'indices':
            columns = message.value

            setup()

            # riceve l'update dal nodo Cloud e lo aggiorna localmente
            t1 = Thread(target=update_model_remote, args=())
            print("Thread update_model_remote avviato")
            # consuma i dati dai sensori
            t2 = Thread(target=consume_sensor_data, args=())
            print("Thread consume_sensor_data avviato")
            # aggiorna il modello localmente e lo invia al nodo Cloud
            t3 = Thread(target=update_model_local, args=())
            print("Thread update_model_local avviato")

            t1.start()
            t2.start()
            t3.start()

            t1.join()
            print("Thread aggiornamento remoto terminato")
            t2.join()
            print("Thread consumo dati terminato")
            t3.join()
            print("Thread aggiornamento locale terminato")
        else:
            print(BColors.FAIL + "Indici assenti" + BColors.ENDC)
    else:
        print(BColors.FAIL + "Modello assente" + BColors.ENDC)


if __name__ == "__main__":
    main()
