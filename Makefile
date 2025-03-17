DOCKER_USER=josecaloca
REPO_NAME=multiclass-text-classification

API_TAG=$(DOCKER_USER)/$(REPO_NAME):api
DATA_PREP_TAG=$(DOCKER_USER)/$(REPO_NAME):data-prep
MODEL_TRAIN_TAG=$(DOCKER_USER)/$(REPO_NAME):train

.PHONY: all build push run clean help

all: build push

############################################################
## Build all Docker images
############################################################
build: build-api build-data-prep build-model-train

build-api:
	cd services/api && docker build -f Dockerfile -t model_api .
	docker tag model_api $(API_TAG)

build-data-prep:
	cd services/data_preparation && docker build -f Dockerfile -t data_preparation .
	docker tag data_preparation $(DATA_PREP_TAG)

build-model-train:
	cd services/model_training && docker build -f Dockerfile -t model_training .
	docker tag model_training $(MODEL_TRAIN_TAG)

############################################################
## Push all Docker images to Docker Hub
############################################################
push: push-api push-data-prep push-model-train

push-api:
	docker push $(API_TAG)

push-data-prep:
	docker push $(DATA_PREP_TAG)

push-model-train:
	docker push $(MODEL_TRAIN_TAG)

############################################################
## Run Docker containers
############################################################
run: run-api run-data-prep run-model-train

run-api:
	docker run --rm --name api-container -it -p 8000:8000 $(API_TAG)

run-data-prep:
	docker run --rm --name data-prep-container -it $(DATA_PREP_TAG)

run-model-train:
	docker run --rm --name model-train-container -it $(MODEL_TRAIN_TAG)

############################################################
## Stop and remove running containers
############################################################
stop:
	docker stop api-container data-prep-container model-train-container || true
	docker rm api-container data-prep-container model-train-container || true

############################################################
## Clean up locally built images
############################################################
clean: stop
	docker rmi model_api data_preparation model_training || true
	docker rmi $(API_TAG) $(DATA_PREP_TAG) $(MODEL_TRAIN_TAG) || true

############################################################
## Show available Makefile commands
############################################################
help:
	@echo "Makefile commands:"
	@echo "  make build     - Build all Docker images"
	@echo "  make push      - Push all images to Docker Hub"
	@echo "  make run       - Run all Docker containers"
	@echo "  make stop      - Stop and remove running containers"
	@echo "  make clean     - Remove all locally built images"
	@echo "  make all       - Build and push all images"
	@echo "  make help      - Show this help message"
