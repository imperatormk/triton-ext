# Shortcuts for building the entire project; each extension is built
# independently.

BUILD_DIR ?= build

default: build

.PHONY: list
list:
	ci/list_extensions.py

.PHONY: build
build:
	@for EXT_DIR in $(shell ci/list_extensions.py path); do \
		$(MAKE) -C $$EXT_DIR build || exit $$?; \
	done

.PHONY: install
install:
	@for EXT_DIR in $(shell ci/list_extensions.py path); do \
		$(MAKE) -C $$EXT_DIR install || exit $$?; \
	done

.PHONY: test
test:
	@FAILED=""; \
	for EXT_DIR in $(shell ci/list_extensions.py path); do \
		$(MAKE) -C $$EXT_DIR test || FAILED="$$FAILED $$EXT_DIR"; \
	done; \
	pytest testing || FAILED="$$FAILED testing"; \
	if [ -n "$$FAILED" ]; then \
		echo "FAILED:$$FAILED"; \
		exit 1; \
	fi

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)

.PHONY: clean-all
clean-all: clean
	rm -rf triton-* llvm-*
