.PHONY: build-image push-image run-image

DOCKER_HUB_REPOSITORY := $(DOCKER_HUB_REPOSITORY)
ENFOBENCH_VERSION := 0.3.1
IMAGE_NAME := sf-naive
IMAGE_TAG := $(ENFOBENCH_VERSION)-$(IMAGE_NAME)
PORT := 3000

image:
	docker build -t $(DOCKER_HUB_REPOSITORY):$(IMAGE_TAG) ./models/$(IMAGE_NAME)

push-image: image
	docker push $(DOCKER_HUB_REPOSITORY):$(IMAGE_TAG)

run-image: image
	docker run -it --rm -p $(PORT):3000 $(DOCKER_HUB_REPOSITORY):$(IMAGE_TAG)
