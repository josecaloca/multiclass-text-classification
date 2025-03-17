model-api:
	uv run app.py

build:
	docker build -f Dockerfile -t model_api .

run: build
	docker run --rm --name model_api_container -it -p 8000:8000 model_api # added port mapping to port 8000 from the contanier on my host (local computer).

clean:
	docker rm -f model_api_container || true
	docker image rm model_api || true