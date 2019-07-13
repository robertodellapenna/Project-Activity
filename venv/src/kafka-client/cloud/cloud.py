import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, chi2
from kafka import KafkaProducer, KafkaConsumer
import pickle
import json
import sys
import datetime
from sklearn.metrics import accuracy_score, recall_score


class BColors:
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'


def main():
    # leggo i csv
    scania_train_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00421/aps_failure_training_set.csv'
    scania_test_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00421/aps_failure_test_set.csv'
    scania_train = pd.read_csv(scania_train_url, header=14, na_values='na')
    scania_test = pd.read_csv(scania_test_url, header=14, na_values='na')

    # sostituisco i missing values con la media della colonna
    scania_train.fillna(scania_train['ab_000'].mean(), inplace=True)
    scania_test.fillna(scania_test['ab_000'].mean(), inplace=True)

    # sostituisco i valori della classe con interi
    scania_train['class'].replace(to_replace=dict(pos=1, neg=0), inplace=True)
    scania_test['class'].replace(to_replace=dict(pos=1, neg=0), inplace=True)

    # converto tutti i float in interi
    scania_train = scania_train.astype('int64')
    scania_test = scania_test.astype('int64')

    # separo attributi e classe
    features_train = scania_train.drop('class', axis=1)
    labels_train = scania_train['class']
    features_test = scania_test.drop('class', axis=1)
    labels_test = scania_test['class']

    # filtro i 90 attributi più significativi e le prime 2k istanze per il train
    indices = SelectKBest(chi2, k=90).fit(features_train, labels_train).get_support(indices=True)
    features_train = pd.DataFrame(features_train).take(indices, axis='columns').head(2000)
    labels_train = labels_train.head(2000)
    features_test = pd.DataFrame(features_test).take(indices, axis='columns')

    # costruisco il modello iniziale
    nb_model = GaussianNB()
    nb_model.fit(features_train, labels_train)

    # calcolo l'accuratezza
    nb_out = nb_model.predict(features_test)
    nb_accuracy = accuracy_score(labels_test, nb_out)
    nb_recall = recall_score(labels_test, nb_out, pos_label=1)
    nb_specificity = recall_score(labels_test, nb_out, pos_label=0)
    print("Accuracy, Recall and Specificity: %s, %s, %s" % (nb_accuracy, nb_recall, nb_specificity))

    # riunisco attributi e classe
    scania_train = features_train
    columns = scania_train.columns.values
    scania_train.insert(0, 'class', labels_train)

    producer = KafkaProducer(bootstrap_servers=['localhost:9092', 'localhost:9093'])

    # invio il modello al fog
    serialized = pickle.dumps(nb_model)
    producer.send('cloud-fog', key=b"model", value=serialized, partition=0)
    producer.send('cloud-fog', key=b"model", value=serialized, partition=1)

    serialized = pickle.dumps(columns)
    producer.send('cloud-fog', key=b"indices", value=serialized, partition=0)
    producer.send('cloud-fog', key=b"indices", value=serialized, partition=1)
    print("Modello inziale e colonne inviati al fog")
    producer.flush()

    consumer = KafkaConsumer('fog-cloud',
                             group_id='model_update',
                             bootstrap_servers=['localhost:9092', 'localhost:9093'])

    print("In attesa del modello dal fog..")
    # ricevo il modello dal fog
    for message in consumer:
        key = message.key.decode("utf-8")
        if key == "model":
            # deserializzo e estraggo il valore
            model = pickle.loads(message.value)
            t = datetime.datetime.now()
            print(BColors.OKBLUE + str(t) + " - Modello ricevuto dal fog, peso " + BColors.ENDC)

            # calcolo l'accuratezza
            out = model.predict(features_test)
            accuracy = accuracy_score(labels_test, out)
            recall = recall_score(labels_test, out, pos_label=1)
            specificity = recall_score(labels_test, out, pos_label=0)
            print(BColors.OKBLUE + "Accuracy, Recall e Specificity: " + str(accuracy) + ", " + str(recall) + ", " + str(specificity) + BColors.ENDC)
            differences = [accuracy - nb_accuracy, recall - nb_recall, specificity - nb_specificity]
            # print(BColors.OKBLUE + "Differenze: " + str(differences) + BColors.ENDC)

            message = next(consumer)

            # se il miglioramento è consistente propago il nuovo modello a tutti i nodi fog
            if accuracy > 0.95 and recall > 0.75 and specificity > 0.95:
                if differences[1] > 0:
                    key = message.key.decode("utf-8")
                    if key == "train":
                        # differenza tra train set iniziale e attuale
                        train = (np.frombuffer(message.value, dtype='int64'))

                        num_cols = int(len(columns) + 1)
                        num_rows = int(len(train) / num_cols)

                        new_train = pd.DataFrame(train.reshape([num_rows, num_cols]), columns=np.append(['class'], columns))

                        serialized = pickle.dumps(model)
                        t = datetime.datetime.now()
                        print(BColors.OKGREEN + str(t) + " - Update modello" + BColors.ENDC)
                        producer.send('cloud-fog', key=b"model", value=serialized, partition=0)
                        producer.send('cloud-fog', key=b"model", value=serialized, partition=1)
                        print("Modello inviato al fog..\n")

                        serialized = new_train.to_numpy().tobytes()
                        producer.send('cloud-fog', key=b"train", value=serialized, partition=0)
                        producer.send('cloud-fog', key=b"train", value=serialized, partition=1)
                        print("Istanze di train inviate al fog\n")
                        producer.flush()

                        nb_accuracy = accuracy
                        nb_recall = recall
                        nb_specificity = specificity
                    else:
                        print(BColors.FAIL + "Assenza istanze di train" + BColors.ENDC)
                else:
                    print(BColors.WARNING + "Assenza di miglioramento" + BColors.ENDC)
            else:
                print(BColors.WARNING + "Accuracy non sufficiente" + BColors.ENDC)
        else:
            print(BColors.FAIL + "Assenza modello" + BColors.ENDC)


if __name__ == "__main__":
    main()
