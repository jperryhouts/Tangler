
%.tsv: %.JPEG
	raveler -f tsv "$<" | awk '/^[0-9]/{print($$1)}' > "$@"

TARGETS=$(shell find train val -name '*JPEG' | sed -e 's/JPEG$$/tsv/')

.PHONY: targets prep train

targets: $(TARGETS)

prep:
	python3 main.py prep ./train ./dataset/train
	python3 main.py prep ./val ./dataset/val

train:
	python3 main.py train ./dataset/
