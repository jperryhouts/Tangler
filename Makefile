
targets=$(shell find train val -name '*JPEG' | sed -e 's/JPEG$$/tsv/')

%.tsv: %.JPEG
	raveler -f tsv "$<" | awk '/^[0-9]/{print($$1)}' > "$@"

targets: $(targets)