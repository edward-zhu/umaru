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
- [utf8](https://github.com/clementfarabet/lua-utf8)

## Build

```sh
$ ./build.sh
```

## Usage

- You could modify the settings in the `main.lua` directly and execute `th main.lua`, the input format is clstm-like (`.png` and `.gt.txt` pair) and you should put all input file path in a text file.
- or if you prefer to use a JSON-format configuration file, you could following the example shows below, and run:

```sh
$ th main.lua -setting [setting file]
```

There would be a folder created in the `experments` folder for every experiment. You could check out the log, settings and saved models there.

## Example Configuration File

descriptions for each option could be found in `main.lua`.

```js
{
  "project_name": "uy_rbm_noised",
  "raw_input": false,
  "hidden_size": 200,
  "nthread": 3,
  "clamp_size": 1,
  "ctc_lua": false,
  "recurrent_unit": "gru",
  "test_every": 2000,
  "omp_threads": 1,
  "show_every": 10,
  "testing_list_file": "wwr.txt",
  "input_size": 48,
  "testing_ratio": 1,
  "max_param_norm": false,
  "training_list_file": "full-train.txt",
  "feature_size": 240,
  "momentum": 0.9,
  "dropout_rate": 0.5,
  "max_iter": 10000000000,
  "save_every": 10000,
  "learning_rate": 0.0001,
  "stride": 5,
  "gpu": false,
  "rbm_network_file": "rbm/wwr.rbm",
  "windows_size": 10
}
```


## LICENSE

BSD 3-Clause License