.PHONY: build-image run-image

image:
	docker build -t $(IMAGE_NAME):latest ./models/$(IMAGE_NAME)


run-image: image
	docker run -it --rm -p 3000:3000 $(IMAGE_NAME):latest
