import os

import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler


class CustomDataLoader:
    def __init__(self) -> None:
        pass

    def get_dataset(self, filepath: str):
        building_id = os.path.basename(filepath).split(".")[0].split("-")[0]
        if "csv" in filepath:
            elec = pd.read_csv(filepath, index_col=1)
        elif "parquet" in filepath:
            elec = pd.read_parquet(filepath)
            elec["timestamp"] = pd.to_datetime(
                elec["timestamp"], format="%Y-%m-%d %H:%M:%S"
            )
            datetime = elec[["timestamp"]]
            elec.set_index("timestamp", inplace=True, drop=True)
        else:
            print("Incompatible filetype")
            return None, None, None, None

        timeseries = elec[["out.site_energy.total.energy_consumption"]].values.astype(
            "float32"
        )
        datetime = datetime.to_numpy()
        return building_id, timeseries, datetime, elec

    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]

        df = pd.DataFrame(data)
        cols, names = list(), list()

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

    def get_train_test_dataset_loaders(
        self, batch_size=50, dataset_path=None, args: dict = None
    ):
        lookback = args["lookback"]
        normalize_data = args["normalize_data"]
        pred_forward = args["pred_forward"]

        id, timeseries, date_time, df = self.get_dataset(dataset_path)

        train_loader = None
        test_loader = None

        if id:
            if normalize_data:
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled = scaler.fit_transform(timeseries)
            else:
                scaled = timeseries

            reframed = self.series_to_supervised(scaled, lookback, pred_forward)
            timestamps = self.series_to_supervised(date_time, lookback, pred_forward)

            timeseries = reframed.values
            tn = timestamps.values

            # print(timeseries.shape)
            # print(tn.shape)

            train_size = int(len(timeseries) * 0.67)
            test_size = len(timeseries) - train_size

            train, test = timeseries[:train_size], timeseries[train_size:]
            tn_train, tn_test = tn[:train_size], tn[train_size:]

            n_mins = lookback
            n_features = 1

            n_obs = n_mins * n_features
            train, test = torch.tensor(train), torch.tensor(test)

            train_X, train_y = train[:, :n_obs], train[:, -pred_forward:]
            test_X, test_y = test[:, :n_obs], test[:, -pred_forward:]
            train_X = train_X.reshape((train_X.shape[0], n_mins, n_features))
            test_X = test_X.reshape((test_X.shape[0], n_mins, n_features))

            print(
                "Train, test shapes:",
                train_X.shape,
                train_y.shape,
                test_X.shape,
                test_y.shape,
            )

            tn_train_X, tn_train_y = tn_train[:, :n_obs], tn_train[:, -pred_forward:]
            tn_test_X, tn_test_y = tn_test[:, :n_obs], tn_test[:, -pred_forward:]
            tn_train_X = tn_train_X.reshape((tn_train_X.shape[0], n_mins, n_features))
            tn_test_X = tn_test_X.reshape((tn_test_X.shape[0], n_mins, n_features))

            train_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(train_X, train_y),
                shuffle=False,
                batch_size=batch_size,
            )
            test_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(test_X, test_y),
                shuffle=False,
                batch_size=batch_size,
            )
            # print("train_X = ", train_X)
            # print("train_y = ", train_y)
            print(
                "LSTM.loader.CustomDataLoader.get_train_test_dataset_loaders:: Dataloaders generated"
            )

        return train_loader, test_loader
