# umaru
An OCR-system based on torch using the technique of LSTM/GRU-RNN, CTC and referred to the works of rnnlib and clstm.

## Notice

This work is now completely INSTABLE, EXPERIMENTAL and UNDER DEVELOPMENT.

For some reason it **can not reach the correct rate of clstm**, I am trying to figure it out (if u have any idea, PLEASE tell me.)

Nevertheless, it might be a good start if you are new to torch or RNN.

## Dependencies

- [torch](https://github.com/torch/torch7) (and following package)
- image
- nn/cunn
- optim
- [rnn](https://github.com/Element-Research/rnn)
- [json](https://github.com/clementfarabet/lua---json)
- utf8 (I am not sure which version I installed, so just type `luarocks install utf8`)

## Usage

- You can just change the settings in the `main.lua`, the input format is clstm-like (`.png` and `.gt.txt` pair) and you should put all input file path in a text file.
- or if you want to choose to use a JSON-format configuration file. An example shows below, and you can use it by execute.

```sh
$ th main -setting [setting file]
```

## Example Configuration File

```json
{
    "clamp_size": 1,
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
    "recurrent_unit": "gru",
    "save_every": 10000,
    "show_every": 10,
    "test_every": 1000,
    "testing_ratio": 1,
    "training_list_file": "wwr.txt"
}
```
