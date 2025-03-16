data-preparation-pipeline:
	uv run pipeline.py

build:
	docker build -f Dockerfile -t data_preparation .

run: build
	docker run --rm --name data_pipeline_container -it data_preparation

clean:
	docker rm -f data_pipeline_container || true
	docker image rm data_preparation || true