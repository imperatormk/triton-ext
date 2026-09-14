WHEEL_DIR ?= dist
AGPU_BUILD_DIR ?= agpu/build

.PHONY: build
build:
	pip wheel . --no-build-isolation --no-deps --wheel-dir $(WHEEL_DIR)

.PHONY: install
install:
	pip install $(WHEEL_DIR)/*.whl

.PHONY: test
test:
	cmake -S agpu -B $(AGPU_BUILD_DIR)
	cmake --build $(AGPU_BUILD_DIR)
	ctest --test-dir $(AGPU_BUILD_DIR) --output-on-failure
	pytest
