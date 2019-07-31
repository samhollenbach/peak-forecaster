import loadimport processimport forecasterfrom pprint import pprintimport numpy as npimport pandas as pdimport seaborn as snsfrom display import baseline_plot, plot_historyimport matplotlib.pyplot as pltnon_solar_sites = ['WM3138', 'WM3708', 'WM5606', 'WM5859', 'WM4132']solar_sites = ['WM3140', 'WM3796', 'WM5603', 'WM5635']data = load.load_all_data(solar_sites[:1])print("\n##### Finished Loading Data ##### \n")# print(data.isna().sum())print("Columns: ", list(data.columns))print(data.describe())print(data.tail())# sns.pairplot(data, diag_kind="kde")# plt.show()print("\n##### Splitting Train/Test Data #####\n")x_train_data, y_train_data, x_test_data, y_test_data = load.train_test_split(data, seed=3)x_train, y_train, x_test, y_test = process.extract_features(x_train_data, y_train_data, x_test_data, y_test_data)print("##### Train/Test Set Size #####")print("Train: ", len(x_train))print("Test: ", len(x_test))print("\n##### Example Features #####")print("Train X:", x_train[-1])print("Train Y:", y_train[-1])print("Test X:", x_test[-1])print("Test Y:", y_test[-1], '\n')################ Build Model ################input_shape = len(x_train[0])output_shape = len(y_train[0])model = forecaster.build_model(input_shape, output_shape)model.summary()################ Train Model ################history = forecaster.train_model(model, x_train, y_train,                                 x_test, y_test)plot_history(history)################### Evaluate Model ###################results = model.evaluate(x_test, y_test, batch_size=24)depth_results = []for test_index in range(len(x_test)):    test_pred, test_time  = model.predict(np.array([x_test[test_index]]))[0]    test_true, time_true = y_test[test_index]    dif = test_true - test_pred    depth_results.append({'true':test_true,                          'pred': test_pred,                          'absolute_error': dif,                          'percentage_error':  dif / test_true,                          })    show_plots = True    if show_plots:        if abs(dif) > 20:            baseline_plot(x_test_data[test_index], y_test_data[test_index], test_pred, test_time)absolute_error = [d['absolute_error'] for d in depth_results]percentage_error = [d['percentage_error'] for d in depth_results]true_values = [d['true'] for d in depth_results]MAE = np.mean(np.abs(absolute_error))MAPE = np.mean(np.abs(percentage_error))abs_std = np.std(absolute_error)max_err_pos = np.max(absolute_error)max_err_neg = np.min(absolute_error)print("##### Mean Absolute Error #####")print(f"{MAE:.4f}")print("##### Mean Absolute Percentage Error #####")print(f"{MAPE:.4f}")print("##### Absolute Error STD #####")print(f"{abs_std:.4f}")print("##### Max Error Positive #####")print(f"{max_err_pos:.4f}")print("##### Max Error Negative #####")print(f"{max_err_neg:.4f}")# plt.plot([i for i in range(len(dif))], dif)plt.scatter(true_values, absolute_error)plt.show()plt.hist(absolute_error, bins=16)plt.show()# plt.hist(percentage_error, bins=16)plt.show()