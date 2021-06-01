# Tangler: Data Prep

Training this model requires that all images be in a common size/shape, and that each one has already been converted into a string sequence using the traditional algorithm. There are two primary data preparation steps which need to be done before we can train the model:

1. Generate training examples from each image (normalize the image, and generate a string pattern)
2. Concatenate examples into `.tfrecord` files

In principle these can all be done in one step, but it helps to break the task down into stages.

## Stage 0: Setup containerized environment

The data preprocessing container can be configured on an EC2 instance as follows:

```bash
sudo yum install git docker

sudo service docker start

sudo usermod -a -G docker ec2-user
# [log out / in]

git clone https://github.com/jperryhouts/Tangler.git

cd Tangler/DataPrep

docker build -t tangler/data_prep .
```

Once the image is built, start an interactive terminal session:

```bash
docker run -it --rm -v "$HOME/Data:/DATA" tangler/data_prep bash
```

## Stage 1: Generate training examples

Processing each individual image takes under a second, but on a dataset with over a million samples this can become very time consuming. The algorithm does not need much RAM, but access to a large number of cores is essential. With the container setup as described above, convert all images into string patterns:


```bash
find /DATA -name '*JPEG' | while read fn ; do
    if [ ! -f "${fn%.*}.raveled" ]; then echo "$fn" ; fi
done | parallel -j200 bash do_ravel.sh {}
```

For each image, `$HOME/Data/**/*.JPEG`, this will generate a normalized version: `$HOME/Data/**/*.jpg`, and a target string sequence: `$HOME/Data/**/*.raveled`. Note that I am over-allocating resources by a lot, hoping that will prompt the operating system to do a lot of IO operations in the background and saturate the CPU resources (this was run on an EC2 instance with 32 ARM cores). In my case I was able to process about 2200 images per minute, but you might get more mileage out of requesting fewer parallel processes.

Once you're certain that the above command completed successfully, you can delete the original images:

```bash
find /DATA -name '*JPEG' -exec rm "{}" \;
```

## Stage 2: Concatenate examples into .tfrecord files

One major bottleneck in training neural networks with large datasets is file access. Network filesystems are especially ill-equipped to deal with large numbers of small files. A major speed-up can be gained by concatenating many individual examples into larger files. This is done by `make_tfrecords.py`, from within the same container:

```bash
python3 make_tfrecords.py "/DATA/train" "/DATA/tfrecords/train"
```
