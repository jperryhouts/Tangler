# Tangler

Tangler is a real-time video filter, inspired by physical [string art](https://vimeo.com/175653201). It uses a neural net architecture to create stylized representations of arbitrary input images. It can be used as a webcam filter for video chats, or convert video clips into string path renderings. There is also an [in-browser live demo](https://jperryhouts.github.io/Tangler/) of the algorithm.

https://user-images.githubusercontent.com/6118266/123179136-930b0a80-d43d-11eb-972a-e515f7eef12d.mp4

[Full demo video](https://storage-9iudgkuqwurq6.s3-us-west-2.amazonaws.com/tangler_480.mp4) \[shark content from [Discovery UK](https://www.youtube.com/watch?v=pgYmY6--DjI)\]

A complete discussion of the model design and implementation can be found in [the documentation](docs/ABOUT.md). This repository contains all code necessary for training, testing, and deploying the Tangler model.

## Installation

Via git:

```bash
git clone https://github.com/jperryhouts/Tangler.git
cd Tangler
pip install -r requirements.txt

python3 -m tangler --help
```

## Quick start

In order to use the pre-trained model, you'll have to download it separately

```bash
wget https://storage-9iudgkuqwurq6.s3-us-west-2.amazonaws.com/tangler_model.h5
```

To run inferences in real time, use Tangler's demo mode:

```bash
python3 -m tangler demo tangler_model.h5
```

If you're on Linux, the model can pipe its output to a virtual webcam device. You must first have [pyfakewebcam](https://github.com/jremmons/pyfakewebcam) installed and properly configured.

```bash
python3 -m tangler demo --webcam /dev/video1 tangler_model.h5
```

## Usage

Tangler is usually run by executing the module directly. For instance, from within the root folder of the Tangler repository, \``python3 -m tangler --help`\` prints the following help message:

```text
usage: python3 -m tangler [RUNTIME ARGUMENTS]

optional arguments:
  -h, --help            show this help message and exit
  --debug               Enable TensorFlow debug mode
  --cpu                 Disable GPU compute

Mode:
  Which action to perform.

  {train,evaluate,demo,convert}
    train               Train the model
    evaluate            Evaluate metrics on test data
    demo                Run inferences in demo mode
    convert             Convert model for tflite or tfjs
```

Each of the operating modes has its own options, which can be shown using the corresponding mode argument, e.g.

```bash
python3 -m tangler demo --help
```

The full command will look something like:

```bash
python3 -m tangler --cpu demo --mirror --stats tangler_model.h5
```

## Testing

Test coverage is still pretty sparse, but there are a handful of unit tests for some of the simple data manipulation functions. They can be run with pytest.

```bash
python -m pytest tangler/
```
