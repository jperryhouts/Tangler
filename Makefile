## Shell variable ${DATA} must be defined
## e.g.:
##    DATA="$HOME/Data/imagenet" make -j8 ravel

PATTERNS=$(shell find $(DATA) -name '*JPEG' | sed -e 's/JPEG$$/raveled/')
NORMALIZED=$(shell find $(DATA) -name '*JPEG' | sed -e 's/JPEG$$/_norm.jpg/')

.PHONY: ravel norm

ravel: $(PATTERNS)

norm: $(NORMALIZED)

%_norm.jpg: %.JPEG
	convert "$<" -gravity center -extent 1:1 -resize 300x300 -grayscale Rec709Luma -format jpeg "$@"

%.raveled: %.JPEG
	convert "$<" -gravity center -extent 1:1 -resize 300x300 -grayscale Rec709Luma -format jpeg - \
		| tee "$*_norm.jpg" | convert - -depth 8 GRAY:- \
		| raveler -f tsv -N 6000 -k 256 -w 100e-6 -r 300 - \
		| awk '/^[0-9]/{print($$1)}' > "$@"
	@if [ `du -s "$@" | cut -f 1` -eq 0 ]; then echo "Failed <$@>"; rm "$@"; fi

prep:
	python3 tangler.py prep ./data/train ./data/tfrecords/train
	python3 tangler.py prep ./data/val ./data/tfrecords/val

%.model:
	python3 tangler.py train -o "$@" ./data/tfrecords/train/*tfrecord
