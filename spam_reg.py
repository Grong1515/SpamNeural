from logreg import LogisticRegression, LogRegError
import numpy


RESULTS_PATH = 'results/'


def load_data(cross_valid_parts_count, split=3000):
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

    one_part = int(len(answer_array)/cross_valid_parts_count)
    learn_parts = []
    answer_parts = []
    for i in range(cross_valid_parts_count):
        learn_parts.append(data_array[cross_valid_parts_count*i:cross_valid_parts_count*(i+1)])
        answer_parts.append(answer_array[cross_valid_parts_count*i:cross_valid_parts_count*(i+1)])

    X_train = numpy.array(learn_parts)
    y_train = numpy.array(answer_parts)

    X_test = numpy.array(data_array[split:])
    y_test = numpy.array(answer_array[split:])
    return (X_train, y_train), (X_test, y_test)


def check(results, test_set):
    n_errors = numpy.sum(results != test_set)
    print('Errors on test set: {0:%}'.format(float(n_errors) / float(n_test_samples)))


if __name__ == '__main__':
    import os.path
    k = 5
    train_set, test_set = load_data(k)
    print(train_set[1].sum())
    classifier_name = 'log_reg_for_spam_1.txt'
    classifier = LogisticRegression()
    if os.path.exists(RESULTS_PATH + classifier_name):
        classifier.load(RESULTS_PATH + classifier_name)
    else:
        for i in range(k):
            for j in range(k):
                if j != i:
                    classifier.fit(train_set[0][j], (train_set[1][j] == 1).astype(numpy.float), 0.0001, 100, 400)
            # print(check(classifier.predict(test_set[0][i]), test_set[1][i]))
        classifier.save(RESULTS_PATH + classifier_name)
    n_test_samples = test_set[1].shape[0]
    results = classifier.predict(test_set[0])
    check(results, test_set[1])
