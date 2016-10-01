from logreg import LogisticRegression, LogRegError
import numpy


learn_data_start = 2300
learn_data_train = 400
learn_data_test = 400
RESULTS_PATH = 'results/'


def load_data():
    f = open('spambase/spambase.data', 'r')
    base_text_data = f.read()

    data_string_array = base_text_data.split('\n')
    data_array = [list(map(lambda a: float(a), string.split(','))) for string in data_string_array if string != '']

    data_array = data_array[learn_data_start:learn_data_start+learn_data_train+learn_data_test:]

    # print(data_array)
    prop_array = []
    answer_array = []

    for vector in data_array:
        answer_array.append(vector.pop(-1))
        prop_array.append(vector)

    X_train = numpy.array(data_array[:learn_data_train])
    y_train = numpy.array(answer_array[:learn_data_train])

    X_test = numpy.array(data_array[learn_data_train:])
    y_test = numpy.array(answer_array[learn_data_train:])
    return (X_train, y_train), (X_test, y_test)


if __name__ == '__main__':
    import os.path
    train_set, test_set = load_data()
    classifiers = list()
    classifier_name = 'log_reg_for_spam_1.txt'
    new_classifier = LogisticRegression()
    if os.path.exists(RESULTS_PATH + classifier_name):
        new_classifier.load(RESULTS_PATH + classifier_name)
    else:
        new_classifier.fit(train_set[0], (train_set[1] == 0).astype(numpy.float))
        new_classifier.save(RESULTS_PATH + classifier_name)
    classifiers.append(new_classifier)
    print((train_set[1] == 0).astype(numpy.float))
    # на тестовом множестве вычисляем результаты распознавания цифр коллективом из 10 обученных логистических регрессий
    # (принцип принятия решений таким коллективом: входной вектор признаков считается отнесённым к тому классу, чья
    # логистическая регрессия выдала наибольшую вероятность).
    n_test_samples = test_set[1].shape[0]
    outputs = numpy.empty((n_test_samples, 10), dtype=numpy.float)
    outputs[:, 0] = classifiers[0].transform(test_set[0])
    results = outputs.argmax(1)
    print(results)
    # сравниваем полученные результаты с эталонными и оцениваем процент ошибок коллектива логистических регрессий
    n_errors = numpy.sum(results != test_set[1])
    print('Errors on test set: {0:%}'.format(float(n_errors) / float(n_test_samples)))
