WHEEL_DIR ?= dist

.PHONY: build
build:
	pip wheel . --no-build-isolation --no-deps --wheel-dir $(WHEEL_DIR)

# Uninstalled first because the version does not change between builds, so
# pip would keep the stale wheel. --force-reinstall in its place would refetch
# triton over the pinned build that carries the plugin API.
.PHONY: install
install:
	-pip uninstall -y -q triton-apple-backend
	pip install $(WHEEL_DIR)/*.whl

.PHONY: test
test:
	pytest
