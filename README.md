# umaru
An OCR-system based on torch using the technique of LSTM/GRU-RNN, CTC and referred to the works of rnnlib and clstm.

## Notice

This work is now completely UNSTABLE, EXPERIMENTAL and UNDER DEVELOPMENT.

[UPDATE] It finally works.

Nevertheless, it might be a good start if you are new to torch or RNN.

## Dependencies

- [torch](https://github.com/torch/torch7) (and following packages)
- image
- nn/cunn
- optim
- [rnn](https://github.com/Element-Research/rnn)
- [json](https://github.com/clementfarabet/lua---json)
- utf8 (I am not sure which version I installed, so just type `luarocks install utf8`)

## Usage

- You can change the settings in the `main.lua` directly and execute `th main.lua`, the input format is clstm-like (`.png` and `.gt.txt` pair) and you should put all input file path in a text file.
- or if you prefer to use a JSON-format configuration file. An example shows below, and you can use it by execute:

```sh
$ th main.lua -setting [setting file]
```

## Example Configuration File

```js
{
    "clamp_size": 1,    // clip the gradient
    "ctc_lua": false,
    "dropout_rate": 0,
    "gpu": false,
    "hidden_size": 200,
    "input_size": 48,
    "learning_rate": 0.0001,
    "max_iter": 10000000000,
    "max_param_norm": false,
    "momentum": 0.9,
    "nthread": 3,
    "omp_threads": 1,
    "project_name": "GRU_testing_on_wwr",
    "recurrent_unit": "gru",    // valid recurrent_unit is gru, lstm, lstm_nopeephole
    "save_every": 10000,
    "show_every": 10,
    "test_every": 1000,
    "testing_ratio": 1, // how much of testing set do you want to use for validating? (it would be ignored if you have set a seperate testing_list_file) 
    "training_list_file": "wwr.txt", // input training list, 
    "testing_list_file": "test.txt" // (optional) set it if you want to use a seperate testing set, or the testing(validating) set is part of your training set.
}
```
