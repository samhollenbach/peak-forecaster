from peak_forecaster import load, process, operate, display, networkimport datetimeimport randomsolar_sites = ['WM3140', 'WM3796', 'WM5603', 'WM5635']def test_standard_operator():    site = 'WM5635'    start = '2018-08-03 00:00:00-07'    end = '2018-09-01 23:45:00-07'    data = load.load_all_data(['WM5635'])    print("\n##### Finished Loading Data ##### \n")    print(data.isna().sum())    print("Columns: ", list(data.columns))    print(data.describe())    print(data.tail())    print(data['timestamp'])    tstart = datetime.time(13, 0)    tend = datetime.time(23, 0)    operator = operate.StandardOperator(site, start, end)    daily_data = operator.prepare_data(data)    thresholds = [random.randint(220, 240) for _ in daily_data]    data = operator.run_standard_operation(tstart, tend, data, thresholds)    train_operator = operate.StandardOperator(site, '2018-01-02 00:00:00-07', start)    daily_data_train = operator.prepare_data(data)    print(daily_data_train)    net = network.Network({})    # net.build_model(len(x_test[0]), 1)    display.baseline_plot2(data)    # print("\n##### Splitting Train/Test Data #####\n")    # x_train_data, y_train_data, x_test_data, y_test_data = \    #     load.train_test_split(    #     data, seed=3)    #    # x_train, y_train, x_test, y_test = process.extract_features(x_train_data,    #                                                             y_train_data,    #                                                             x_test_data,    #                                                             y_test_data)test_standard_operator()