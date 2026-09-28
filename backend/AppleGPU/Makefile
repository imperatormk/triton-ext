# The name in pyproject.toml, so a rebuild replaces this wheel
# and leaves everything it depends on alone.
WHEEL_NAME := triton-apple-backend
WHEEL_DIR ?= dist
AGPU_BUILD_DIR ?= agpu/build

.PHONY: build
build:
	pip wheel . --no-build-isolation --no-deps --wheel-dir $(WHEEL_DIR)

.PHONY: install
install:
	uv pip install $(WHEEL_DIR)/*.whl --reinstall-package $(WHEEL_NAME)

.PHONY: test
test:
	cmake -S agpu -B $(AGPU_BUILD_DIR)
	cmake --build $(AGPU_BUILD_DIR)
	ctest --test-dir $(AGPU_BUILD_DIR) --output-on-failure
	pytest
