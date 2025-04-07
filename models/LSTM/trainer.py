import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable


class CustomModelTrainer:
    def __init__(self) -> None:
        pass

    # (model, train_loader, test_loader: None, epochs, criterion, optimizer, args:dict)
    def train_model(
        self,
        model,
        results,
        train_loader,
        epochs,
        timeout_s: float,
        loss_func=None,
        optimizer=None,
        device=torch.device("cpu"),
        test_loader=None,
        args: dict = None,
        start_time: float = None,
    ):
        print(
            f"\nLSTM.trainer.CustomModelTrainer.train_model:: model = {model} starting to train\n"
        )

        model.to(device)

        train_X, train_y = train_loader.dataset.tensors
        if test_loader:
            test_X, test_y = test_loader.dataset.tensors

        model.train()
        counter = 0
        clip = args["clip"]
        train_rmse = 0.0
        test_rmse = 0.0

        # if loss_func is None:
        #   loss_func = torch.nn.MSELoss()
        # if optimizer is None:
        #   optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
        loss_func = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        if start_time is None:
            start_time = time.time()

        try:
            for i in range(epochs):
                for inputs, labels in train_loader:
                    counter += 1
                    inputs, labels = inputs.to(device), labels.to(device)
                    model.zero_grad()
                    output = model(inputs)
                    loss = loss_func(output.squeeze(), labels.squeeze().float())
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                    optimizer.step()

                    if time.time() - start_time >= timeout_s:
                        results = {
                            "epochs": counter / len(train_loader),
                            "loss": loss.item(),
                            "test_rmse": test_rmse.__float__(),
                            "train_rmse": train_rmse.__float__(),
                            "total_mini_batches": counter,
                            "time_taken_s": time.time() - start_time,
                        }
                        print(
                            f"\nLSTM.trainer.CustomModelTrainer.train_model:: Timeout!"
                        )
                        print(
                            f"\nLSTM.trainer.CustomModelTrainer.train_model:: Results - {results}\n"
                        )
                        return results
                    results = {
                        "epochs": counter / len(train_loader),
                        "loss": loss.item(),
                        "test_rmse": test_rmse.__float__(),
                        "train_rmse": train_rmse.__float__(),
                        "total_mini_batches": counter,
                        "time_taken_s": time.time() - start_time,
                    }

                model.eval()
                with torch.no_grad():
                    y_pred = model(train_X)
                    train_rmse = np.sqrt(loss_func(y_pred, train_y))
                    if test_loader:
                        y_pred = model(test_X)
                        test_rmse = np.sqrt(loss_func(y_pred, test_y))

                print(
                    "Epoch %d: loss %.8f, train RMSE %.4f, test RMSE %.4f"
                    % (i, loss.item(), train_rmse, test_rmse)
                )
        except Exception as e:
            print(f"\nLSTM.trainer.CustomModelTrainer.train_model:: Exception - {e}\n")

        results = {
            "epochs": counter / len(train_loader),
            "loss": loss.item(),
            "test_rmse": test_rmse.__float__(),
            "train_rmse": train_rmse.__float__(),
            "total_mini_batches": counter,
            "time_taken_s": time.time() - start_time,
        }
        print(f"\nLSTM.trainer.CustomModelTrainer.train_model:: Results - {results}\n")

        return results

    def validate_model(
        self,
        model,
        dataloader,
        device: str = "cpu",
        loss_func=None,
        optimizer=None,
        round_no=None,
        args: dict = None,
    ):
        def get_dataset(path, filename):
            if "csv" in filename:
                elec = pd.read_csv(path + filename, index_col=1)
            elif "parquet" in filename:
                elec = pd.read_parquet(os.path.join(path, filename))
                elec["timestamp"] = pd.to_datetime(
                    elec["timestamp"], format="%Y-%m-%d %H:%M:%S"
                )
                datetime = elec[["timestamp"]]
                elec.set_index("timestamp", inplace=True, drop=True)
            timeseries = elec[
                ["out.site_energy.total.energy_consumption"]
            ].values.astype("float32")
            datetime = datetime.to_numpy()
            # print("Proccessing ", filename)
            return timeseries, datetime, elec

        def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
            n_vars = 1 if type(data) is list else data.shape[1]
            df = pd.DataFrame(data)
            cols, names = list(), list()
            # input sequence (t-n, ... t-1)
            for i in range(n_in, 0, -1):
                cols.append(df.shift(i))
                names += [("var%d(t-%d)" % (j + 1, i)) for j in range(n_vars)]
            # forecast sequence (t, t+1, ... t+n)
            for i in range(0, n_out):
                cols.append(df.shift(-i))
                if i == 0:
                    names += [("var%d(t)" % (j + 1)) for j in range(n_vars)]
                else:
                    names += [("var%d(t+%d)" % (j + 1, i)) for j in range(n_vars)]
            # put it all together
            agg = pd.concat(cols, axis=1)
            agg.columns = names
            # drop rows with NaN values
            if dropnan:
                agg.dropna(inplace=True)
            return agg

        def predict_iteration(net, testX, lookAhead, device, RNN=True):
            testBatchSize = testX.shape[0]
            # print(testX.shape)
            ans = []

            for i in range(lookAhead):
                testX_torch = torch.from_numpy(testX)
                testX_torch = Variable(testX_torch)
                testX_torch = testX_torch.to(device)
                pred = net(testX_torch)
                pred = pred.cpu().data.numpy()
                # else:
                #     pred = pred.data.numpy()
                pred = np.squeeze(pred)
                ans.append(pred)
                # print(pred[-1])
                testX = testX[:, lookAhead:]  # drop the head
                # print(testX.shape)
                # print(pred.shape)
                # print('###############')
                if RNN:
                    pred = pred.reshape((testBatchSize, lookAhead, 1))
                    testX = np.append(
                        testX, pred, axis=1
                    )  # add the prediction to the tail
                else:
                    pred = pred.reshape((testBatchSize, 1))
                    testX = np.append(
                        testX, pred, axis=1
                    )  # add the prediction to the tail

            ans = np.array(ans)
            # print(ans.shape)
            if lookAhead == 1:
                ans = ans.transpose([1, 0])
                # print(ans.shape)
                return ans
            else:
                # print(ans.shape)
                return ans[0]

        def calcMAPE(true, pred, epsion=0.00000001):
            true += epsion
            return np.mean(np.abs((true - pred) / true)) * 100

        def calcSMAPE(true, pred):
            delim = (np.abs(true) + np.abs(pred)) / 2.0
            return np.mean(np.abs((true - pred) / delim)) * 100

        # data_path = "data/OpenEIA-CA/test"
        try:
            data_path = args["test_data_path"]
            file_list = os.listdir(data_path)
        except Exception as e:
            data_path = "/home/fedml/fedml-ng/data/OpenEIA-CA/train"
            file_list = os.listdir(data_path)

        print("Validation data path - ", data_path)

        lookback = 5
        pred_ahead = 2

        results = {}

        device = torch.device(device)
        # model.to(device)
        loss_func = torch.nn.MSELoss()

        for i, file_name in enumerate(file_list):
            timeseries, date_time, df = get_dataset(data_path, file_name)

            building_id = file_name.split("-")[0]

            scalar = MinMaxScaler(feature_range=(0, 1))
            scaled = scalar.fit_transform(timeseries)

            n_mins = lookback
            n_features = 1

            reframed = series_to_supervised(scaled, lookback, pred_ahead)
            timestamps = series_to_supervised(date_time, lookback, pred_ahead)

            timeseries = reframed.values
            tn = timestamps.values

            train_size = int(len(timeseries) * 0.67)
            test_size = len(timeseries) - train_size
            train, test = timeseries[:train_size], timeseries[train_size:]
            tn_train, tn_test = tn[:train_size], tn[train_size:]

            # print('train shape and test shape')

            n_obs = n_mins * n_features
            train, test = torch.tensor(train), torch.tensor(test)
            train_X, train_y = train[:, :n_obs], train[:, -pred_ahead:]
            test_X, test_y = test[:, :n_obs], test[:, -pred_ahead:]
            train_X = train_X.reshape((train_X.shape[0], n_mins, n_features))
            test_X = test_X.reshape((test_X.shape[0], n_mins, n_features))

            # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
            tn_train_X, tn_train_y = tn_train[:, :n_obs], tn_train[:, -pred_ahead:]
            tn_test_X, tn_test_y = tn_test[:, :n_obs], tn_test[:, -pred_ahead:]

            tn_train_X = tn_train_X.reshape((tn_train_X.shape[0], n_mins, n_features))
            tn_test_X = tn_test_X.reshape((tn_test_X.shape[0], n_mins, n_features))

            model.train(False)

            output = model(test_X)
            loss = loss_func(output.squeeze(), test_y.squeeze().float())

            test_X_ = test_X.data.cpu().numpy()
            testY = test_y.data.cpu().numpy()

            testPred = predict_iteration(
                model, test_X_, pred_ahead, device=device, RNN=True
            )

            testPred = scalar.inverse_transform(testPred)
            testY = scalar.inverse_transform(testY)

            MAE = mean_absolute_error(testY, testPred)
            RMSE = np.sqrt(mean_squared_error(testY, testPred))
            MAPE = calcMAPE(testY, testPred)
            SMAPE = calcSMAPE(testY, testPred)

            # if round_no % 1 == 0:

            if pred_ahead == 1:
                testPred = testPred
                testY = testY
            else:
                x = testPred
                testPred = list(testPred[:, 0])
                testPred.extend(list(x[-1, 1:]))
                x = testY
                testY = list(testY[:, 0])
                testY.extend(list(x[-1, 1:]))
                x = tn_test_y
                tn_test_Y = list(tn_test_y[:, 0])
                tn_test_Y.extend(list(x[-1, 1:]))
                # print(tn_test_Y[:5])

            testPred_ = testPred[-96 * 8 :]
            testY_ = testY[-96 * 8 :]
            tn_test_Y_ = tn_test_Y[-96 * 8 :]

            if args["plot_predictions"]:
                try:
                    save_path = f"{args['metrics_dir']}/plots/{building_id}"
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                except Exception as e:
                    print("Exception::", e)

                plt.rcParams.update({"font.size": 12})
                plt.figure(figsize=(12, 12))
                plt.plot(tn_test_Y_, testPred_, "r", label="Predicted")
                plt.plot(tn_test_Y_, testY_, "g", label="Actual")
                plt.legend()
                plt.xticks(rotation=90)

                plt.xlabel("time")
                plt.ylabel("load")
                plt.title(
                    f"Forecast for building-id = {building_id} for round = {round_no}"
                )
                plt.grid(color="black", alpha=0.5, linewidth=0.5)

                plt.savefig(f"{save_path}/round-{round_no}", dpi=300)
                plt.close()

            building_results = {
                "loss": loss.item(),
                "MAE": MAE,
                "RMSE": RMSE,
                "MAPE": MAPE,
                "SMAPE": SMAPE,
            }

            results[f"{building_id}"] = building_results

        if args["dump_metrics"]:
            save_path = f"{args['metrics_dir']}/metrics"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            with open(f"{save_path}/metrics_{round_no}.bin", "wb") as f:
                pickle.dump(results, f)

        output_results = dict()
        try:
            output_results["individual_loss"] = [
                results[x]["loss"] for x in results.keys()
            ]
            output_results["loss"] = max(output_results["individual_loss"])
            output_results["MAE"] = [results[x]["MAE"] for x in results.keys()]
            output_results["RMSE"] = [results[x]["RMSE"] for x in results.keys()]
            output_results["MAPE"] = [results[x]["MAPE"] for x in results.keys()]
            output_results["SMAPE"] = [results[x]["SMAPE"] for x in results.keys()]

            print("Validation Results = ", output_results)
        except Exception as e:
            print("Exception in LSTM.traner.validate_model::", e)

        print("LSTM.validator:: validation metrics = ", output_results)

        return output_results
