import numpy as np


def evaluate_basic(network, x_train, y_train, x_test, y_test):
    ###############
    # Build Model #
    ###############
    input_shape = len(x_train[0])
    output_shape = len(y_train[0])

    model = network.build_model(input_shape, output_shape)
    model.summary()

    ###############
    # Train Model #
    ###############
    history = network.train_model(model, x_train, y_train)

    ##################
    # Evaluate Model #
    ##################
    # results = model.evaluate(x_test, y_test, batch_size=24)

    depth_results = []
    for test_index in range(len(x_test)):

        test_pred, = model.predict(np.array([x_test[test_index]]))[0]
        test_true, = y_test[test_index]
        dif = test_true - test_pred
        depth_results.append({'true': test_true,
                              'pred': test_pred,
                              'absolute_error': dif,
                              'percentage_error': dif / test_true,
                              })

    absolute_error = [d['absolute_error'] for d in depth_results]
    percentage_error = [d['percentage_error'] for d in depth_results]
    true_values = [d['true'] for d in depth_results]

    MAE = np.mean(np.abs(absolute_error))
    MAPE = np.mean(np.abs(percentage_error))
    abs_std = np.std(absolute_error)
    max_err_pos = np.max(absolute_error)
    max_err_neg = np.min(absolute_error)

    return MAE + max_err_pos + abs_std