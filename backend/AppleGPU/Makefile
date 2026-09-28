# The name in pyproject.toml, so a rebuild replaces this wheel
# and leaves everything it depends on alone.
WHEEL_NAME := triton-apple-backend
WHEEL_DIR ?= dist
AGPU_BUILD_DIR ?= agpu/build

# torch on PyPI depends on a released triton, which replaces the pinned
# nightly, and on the CUDA runtime this never uses. PyTorch's own index serves
# the same package without either. macOS keeps PyPI: its wheel is the one that
# carries MPS, and nothing there pulls triton.
ifeq ($(shell uname),Linux)
TORCH_INDEX := --index pytorch-cpu=https://download.pytorch.org/whl/cpu
endif

.PHONY: build
build:
	pip wheel . --no-build-isolation --no-deps --wheel-dir $(WHEEL_DIR)

.PHONY: install
install:
	uv pip install $(TORCH_INDEX) $(WHEEL_DIR)/*.whl --reinstall-package $(WHEEL_NAME)

.PHONY: test
test:
	cmake -S agpu -B $(AGPU_BUILD_DIR)
	cmake --build $(AGPU_BUILD_DIR)
	ctest --test-dir $(AGPU_BUILD_DIR) --output-on-failure
	pytest
