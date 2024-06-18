PYTHON_ENV = .venv
REQUIREMENTS = requirements.txt
PYTHON_SCRIPT = main.py

venv:
	@echo "creating venv..."
	python3 -m venv $(PYTHON_ENV)
	@echo "done"

install:
	@echo "installing packages..."
	. $(PYTHON_ENV)/bin/activate && pip install -r $(REQUIREMENTS)
	@echo "done"

run:
	@echo "running detector..."
	. $(PYTHON_ENV)/bin/activate && python $(PYTHON_SCRIPT) ${from} ${to}
	@echo "done"
