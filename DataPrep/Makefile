## build and run the docker container
## that will do the actual processing

.PHONY: docker-build docker-run

docker-build:
	docker build -t tangler/data_prep .

docker-run:
	docker run -it --rm -v "$(DATA):/DATA" tangler/data_prep
