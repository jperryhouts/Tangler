# Tangler: Data Prep

Training this model requires that all images be in a common size/shape, and that each one has already been converted into a string sequence using the traditional algorithm. There are two primary data preparation steps which need to be done before we can train the model:

1. Generate training examples from each image (normalize the image, and generate a string pattern)
2. Concatenate examples into `.tfrecord` files

In principle these can all be done in one step, but it helps to break the task down into stages.

## Stage 0: Setup containerized environment

The data preprocessing container can be configured on an EC2 instance as follows:

```bash
sudo yum install git docker

sudo usermod -a -G docker ec2-user
# [log out / in]

git clone https://github.com/jperryhouts/Tangler.git

cd Tangler/DataPrep

docker build -t tangler/data_prep .
```

## Stage 1: Generate training examples

Processing each individual image takes under a second, but on a dataset with over a million samples this can become very time consuming. The algorithm does not need much RAM, but access to a large number of cores is essential. With the container setup as described above, converting all images in the dataset can be done with:

```bash
docker run -it --rm -v "$HOME/Data:/DATA" tangler/data_prep make -j128 ravel
```

For each image, `$HOME/Data/**/*.JPEG`, this will generate a normalized version: `$HOME/Data/**/*_norm.jpg`, and a target string sequence: `$HOME/Data/**/*.raveled`.

## Stage 2: Concatenate examples into .tfrecord files

One of the biggest bottlenecks in training with large datasets is file access. Network filesystems are especially ill-equipped to deal with large numbers of small files. A major speed-up can be gained by concatenating many individual examples into larger files. This is done by `make_tfrecords.py`, from within the same container:

```bash
docker run -it --rm -v "$HOME/Data:/DATA" tangler/data_prep python3 make_tfrecords.py "/DATA/train" "/DATA/tfrecords/train"
```