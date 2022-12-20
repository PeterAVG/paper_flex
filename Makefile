VENV_DIR = .venv

clean:
	rm -rf $(VENV_DIR) poetry.lock venv_julia .tmp .mypy_cache .pytest_cache

clean-tex:
	rm -rf ieee/*.glo ieee/*.aux ieee/*.fls ieee/*.ist ieee/*-glg ieee/*-glo ieee/*-gls ieee/*.synctex.gz ieee/*.dvi ieee/*.log ieee/*.pdf

dev-setup: $(VENV_DIR)/.made
	$(VENV_DIR)/bin/pre-commit install -ft pre-push
	$(VENV_DIR)/bin/pre-commit install -ft pre-commit

$(VENV_DIR)/.made:
	python3 -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install --upgrade pip
	poetry install
	touch $@

julia-setup:
	julia --project=. src/julia/setup.jl "venv_julia"
