## Shell variable ${DATA} must be defined
## e.g.:
##    DATA="$HOME/Data/imagenet" make -j8 ravel

PATTERNS=$(shell find $(DATA) -name '*JPEG' | sed -e 's/JPEG$$/raveled/')
NORMALIZED=$(shell find $(DATA) -name '*JPEG' | sed -e 's/JPEG$$/_norm.jpg/')

.PHONY: ravel norm docker-build docker-run

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

docker-build:
	docker build -t tangler/data_prep .

docker-run:
	docker run -it --rm \
		-v "$(DATA):/DATA" \
		-e username=$(shell whoami) \
		-e userid=$(shell id -u) \
		-e groupid=$(shell id -g) \
		tangler/data_prep