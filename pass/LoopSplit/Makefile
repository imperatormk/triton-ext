WHEEL_NAME := triton-loop-split
WHEEL_DIR ?= dist

.PHONY: build
build:
	pip wheel . --no-build-isolation --no-deps --wheel-dir $(WHEEL_DIR)

.PHONY: install
install:
	uv pip install $(WHEEL_DIR)/*.whl --reinstall-package $(WHEEL_NAME)

.PHONY: test
test:
	pytest
