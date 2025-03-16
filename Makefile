pre-processing-pipeline:
	PYTHONPATH=src uv run -m data_preprocessing.pipeline

fine-tuning-distilbert:
	PYTHONPATH=src uv run -m model_training.distilbert_fine_tuning

train-xgboost:
	PYTHONPATH=src uv run -m model_training.xgboost_trainer

# execute all pipelines sequentially
model-training-execution: pre-processing-pipeline fine-tuning-distilbert train-xgboost

build:
	docker build -f Dockerfile -t ml-pipeline .

run:
	docker run -it ml-pipeline
