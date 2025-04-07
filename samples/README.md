# Samples
This folder contains sample configuration files to train LeNet5, AlexNet, MobileNet and LSTM on their respective datasets. The configuration files serve as a template for users wanting to train their custom model on their custom dataset using Flotilla. Each of the subdirectories (named after the model it is supposed to train) has three confiburation files, namely `client_config.yaml`, `server_config.yaml` and `training_config.yaml`. The first configuration file should be placed in the `config` folder in the root directory on the client side, while the last two should be placed in the same directory, but on the server side. 

Notes:
- All fields mentioned in the configuration files are mandatory.
- If the `aggregator` is set as `default` in the `training_config.yaml`, a simple averaging of all available model weights are performed.
- If the `client_selection` is set as `default` in the `training_config.yaml`, 100% of all available clients are selected every round.

# Model Benchmark Results

To replicate the results as seen below use the configurations provided.

## MobileNet

**Loss versus Accuracy Plot on Server side** 

![MobileNet Loss versus Accuracy Plot on Server side](.results/MobileNet/loss-acc.jpg)
--

**Loss as reported by each client** 

![MobileNet Loss as reported by each client](.results/MobileNet/per-client-loss.jpg)
--

**Accuracy as reported by each client** 

![MobileNet Accuracy as reported by each client](.results/MobileNet/per-client-accuracy.jpg)
--

**Time Taken per round by each client** 

![MobileNet Time Taken per round by each client](.results/MobileNet/per-client-time.jpg)
--

**No. of minibatches ran by each client in Benchmark Phase** 

![MobileNet No. of minibatches ran by each client in Benchmark Phase](.results/MobileNet/per-client-minibatch.jpg)
--

---

## LeNet5

**Loss versus Accuracy Plot on Server side**

![LeNet5 Loss versus Accuracy Plot on Server side](.results/LeNet5/loss-acc.jpg)
--

**Loss as reported by each client**

![LeNet5 Loss as reported by each client](.results/LeNet5/per-client-loss.jpg)
--

**Accuracy as reported by each client**

![LeNet5 Accuracy as reported by each client](.results/LeNet5/per-client-accuracy.jpg)
--

**Time Taken per round by each client**

![LeNet5 Time Taken per round by each client](.results/LeNet5/per-client-time.jpg)
--

**No. of minibatches ran by each client in Benchmark Phase**

![LeNet5 No. of minibatches ran by each client in Benchmark Phase](.results/LeNet5/per-client-minibatch.jpg)
--

---
## AlexNet

**Loss versus Accuracy Plot on Server side**

![AlexNet Loss versus Accuracy Plot on Server side](.results/AlexNet/loss-acc.jpg)
--

**Loss as reported by each client**

![AlexNet Loss as reported by each client](.results/AlexNet/per-client-loss.jpg)
--

**Accuracy as reported by each client**

![AlexNet Accuracy as reported by each client](.results/AlexNet/per-client-accuracy.jpg)
--

**Time Taken per round by each client**

![AlexNet Time Taken per round by each client](.results/AlexNet/per-client-time.jpg)
--

**No. of minibatches ran by each client in Benchmark Phase**

![AlexNet No. of minibatches ran by each client in Benchmark Phase](.results/AlexNet/per-client-minibatch.jpg)
--

---
