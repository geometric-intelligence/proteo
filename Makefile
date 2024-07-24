PYTHON ?= python

.PHONY: test
test:
	pytest -v

.PHONY: autoformat
autoformat:
	black proteo
	isort proteo 

.PHONY: lint
lint:
	$(PYTHON) -m flake8 proteo
	$(PYTHON) -m black proteo --check
	# Note that Bandit will look for .bandit file only if it's invoked with -r option.
	$(PYTHON) -m bandit -c pyproject.toml -r proteo --exit-zero
	$(PYTHON) -m mypy --install-types --non-interactive

.PHONY: clean
clean:
	@find . -type f -name '*.py[co]' -delete -o -type d -name __pycache__ -delete

.PHONY: rmtmp
rmtmp:
	sudo rm -r /tmp && sudo mkdir /tmp && sudo chmod -R 777 /tmp

.PHONY: rmprocessed
rmprocessed:
	rm -rf /home/data/data_louisa/processed/* && sudo chmod -R 777 /home/data/data_louisa/processed


.PHONY: conda-osx-64.lock
conda-osx-64.lock:
	CONDA_SUBDIR=osx-64 conda-lock -f conda.yaml -p osx-64
	CONDA_SUBDIR=osx-64 conda-lock render -p osx-64

.PHONY: conda-linux-64.lock
conda-linux-64.lock:
	conda-lock -f conda.yaml -p linux-64
	conda-lock render -p linux-64

conda-lock.yml: conda-osx-64.lock conda-linux-64.lock

# Clear the cache and rebuild the lock.
.PHONY: poetry.lock
poetry.lock:
	poetry cache clear --all .
	poetry lock