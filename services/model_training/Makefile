model-training-pipeline:
	uv run pipeline.py

build:
	docker build -f Dockerfile -t model_training .

run: build
	docker run --rm --name model_training_container -it model_training

clean:
	docker rm -f data_pipeline_container || true
	docker image rm data_preparation || true	