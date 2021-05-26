
%.tsv: %.JPEG
	raveler -f tsv "$<" | awk '/^[0-9]/{print($$1)}' > "$@"

TARGETS=$(shell find train val -name '*JPEG' | sed -e 's/JPEG$$/tsv/')

.PHONY: targets prep

targets: $(TARGETS)

prep:
	python3 tangler.py prep ./data/train ./data/tfrecords/train
	python3 tangler.py prep ./data/val ./data/tfrecords/val

%.model:
	python3 tangler.py train -o "$@" ./data/tfrecords/train/*tfrecord
