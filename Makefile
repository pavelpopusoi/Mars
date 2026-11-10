.PHONY: install test demo serve lint
install:
	pip install -e ".[dev]"
test:
	pytest -q
serve:
	mars serve
demo:
	mars run "Как микропластик влияет на здоровье человека?" --mock
lint:
	ruff check mars tests
