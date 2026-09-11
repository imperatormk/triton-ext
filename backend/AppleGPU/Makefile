WHEEL_DIR ?= dist

.PHONY: build
build:
	pip wheel . --no-build-isolation --no-deps --wheel-dir $(WHEEL_DIR)

# Reinstalls this wheel over the previous build; a plain --force-reinstall
# would also refetch triton, replacing the pinned plugin-capable one.
.PHONY: install
install:
	pip install $(WHEEL_DIR)/*.whl --force-reinstall --no-deps
	pip install $(WHEEL_DIR)/*.whl

.PHONY: test
test:
	pytest
