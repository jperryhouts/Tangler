# Tangler: Data Prep

The Tangler model was trained on about 1.4 million images from the popular [ImageNet object localization challenge](https://www.kaggle.com/c/imagenet-object-localization-challenge/data). The associated code requires that all images have a corresponding precomputed string path to be used as training targets, and that all images be delivered with a common aspect ratio and color profile. In particular, I chose to train the model on grayscale images with 256x256 pixel resolution, and pin sequences with 6000 line segments. The fully preprocessed dataset can be downloaded [here](https://storage-9iudgkuqwurq6.s3-us-west-2.amazonaws.com/tangled_data_tfrecords.tar), and it can be reproduced with the following steps.

---------

There are two primary data preparation tasks for the tangler model:

1. Generate training examples from each image
    - normalize the image size and format
    - generate a target string pattern
2. Concatenate examples into `.tfrecord` files

In principle these can be done in one step, but it helps to break the task down into stages. The environment for completing these steps is containerized to make it portable between cloud instances. Stage 1 requires substantial computing power, while stage 2 benefits from I/O performance. I chose to switch EC2 instance types between stage 1 and 2 to optimize these operations those metrics, but it's still a slow process when operating on the entire dataset.

## Stage 0: Setup containerized environment

Configure and build the docker image:

```bash
sudo yum install git docker

sudo service docker start

sudo usermod -a -G docker ec2-user
# [log out / in]

git clone https://github.com/jperryhouts/Tangler.git

cd Tangler/DataPrep

docker build -t tangler/data_prep .
```

Spin up a container, and connect to it with an interactive terminal:

```bash
docker run -it --rm -v "$HOME/Data:/DATA" tangler/data_prep bash
```

## Stage 1: Generate training examples

Processing each individual image takes under a second, but on a dataset with over a million samples this can become very time consuming. The algorithm does not need much RAM, but access to a large number of cores is essential. With the container setup as described above, steps 1a and 1b can be processed simultaneously with the `do_ravel.sh` script:

```bash
find /DATA -name '*JPEG' | while read fn ; do
    if [ ! -f "${fn%.*}.raveled" ]; then echo "$fn" ; fi
done | parallel -j200 bash do_ravel.sh {}
```

For each image, `$HOME/Data/**/*.JPEG`, this will generate a normalized version: `$HOME/Data/**/*.jpg`, and a target string sequence: `$HOME/Data/**/*.raveled`. Note that I am over-allocating resources (by a lot), hoping that will prompt the operating system to do a lot of IO in the background and saturate the CPU resources. I ran this on an EC2 instance with 32 ARM cores, and it took several hours to complete. In my case I was able to process about 2200 images per minute.

Once you're certain that the above command completed successfully, you can delete the original images:

```bash
find /DATA -name '*JPEG' -exec rm "{}" \;
```

## Stage 2: Concatenate examples into .tfrecord files

At this point it's possible to train the model using the example generator mode (pass the data directory as the first argument), but training will be extremely slow. Most filesystems are ill-equipped to deal with large numbers of small files. Technically, you'll need to read all those files at some point, so if you were only planning to train once, it might actually be more efficient to read the examples in generator mode. However, if you plan to test different training strategies and hyperparameters, there's [a lot of performance to be gained](https://www.tensorflow.org/tutorials/load_data/tfrecord) by concatenating the individual examples into larger files. This is done by `make_tfrecords.py`, from within the same container:

```bash
python3 make_tfrecords.py "/DATA/train" "/DATA/tfrecords/train"
python3 make_tfrecords.py "/DATA/test" "/DATA/tfrecords/test"
python3 make_tfrecords.py "/DATA/val" "/DATA/tfrecords/val"
```
