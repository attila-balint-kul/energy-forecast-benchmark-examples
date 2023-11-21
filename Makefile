.PHONY: build-image push-image run-image

DOCKER_HUB_REPOSITORY := $(DOCKER_HUB_REPOSITORY)
ENFOBENCH_VERSION := 0.3.2
MODEL := sf-naive
IMAGE_TAG := $(ENFOBENCH_VERSION)-$(MODEL)
PORT := 3000

image:
	docker build -t $(DOCKER_HUB_REPOSITORY):$(IMAGE_TAG) ./models/$(MODEL)

push-image: image
	docker push $(DOCKER_HUB_REPOSITORY):$(IMAGE_TAG)

run-image: image
	docker run -it --rm -p $(PORT):3000 $(DOCKER_HUB_REPOSITORY):$(IMAGE_TAG)
