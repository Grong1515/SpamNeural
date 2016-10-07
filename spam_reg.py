from logreg import LogisticRegression, LogRegError
import numpy


RESULTS_PATH = 'results/'


def load_data():
    f = open('spambase/spambase.data', 'r')
    base_text_data = f.read()

    data_string_array = base_text_data.split('\n')
    numpy.random.shuffle(data_string_array)
    data_array = [list(map(lambda a: float(a), string.split(','))) for string in data_string_array if string != '']


    # print(data_array)
    answer_array = []

    for vector in data_array:
        answer_array.append(vector.pop(-1))

    mean = numpy.mean(data_array, 0)
    std = numpy.std(data_array, 0)

    data_array -= mean
    data_array /= std

    split = 2000
    X_train = numpy.array(data_array[:split])
    y_train = numpy.array(answer_array[:split])

    X_test = numpy.array(data_array[split:])
    y_test = numpy.array(answer_array[split:])
    return (X_train, y_train), (X_test, y_test)


if __name__ == '__main__':
    import os.path
    train_set, test_set = load_data()
    print(train_set[1].sum())
    classifier_name = 'log_reg_for_spam_1.txt'
    classifier = LogisticRegression()
    if os.path.exists(RESULTS_PATH + classifier_name):
        classifier.load(RESULTS_PATH + classifier_name)
    else:
        classifier.fit(train_set[0], (train_set[1] == 1).astype(numpy.float), 0.0001, 100)
        classifier.save(RESULTS_PATH + classifier_name)
    n_test_samples = test_set[1].shape[0]
    results = classifier.predict(test_set[0])
    n_errors = numpy.sum(results != test_set[1])
    print('Errors on test set: {0:%}'.format(float(n_errors) / float(n_test_samples)))
