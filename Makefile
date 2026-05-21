# tensorcore - developer convenience targets.
#
# This Makefile is for human convenience; CI and downstream consumers use
# CMake directly. Don't add anything here that breaks if invoked from a
# non-developer environment.

BUILD_DIR ?= build
BUILD_TYPE ?= Release
INSTALL_PREFIX ?= /tmp/tensorcore-install
WHEEL_DIR ?= /tmp/tc-wheel-out
JOBS ?= 8

PYTHON ?= python3

.PHONY: help build configure clean test bench smoke wheel install \
        check-version check-headers check-exports check-python \
        examples decode train hello inspect \
        docs-check icc-audit all

help:
	@echo "tensorcore - dev shortcuts"
	@echo ""
	@echo "Build:"
	@echo "  make configure        cmake -B $(BUILD_DIR)"
	@echo "  make build            cmake --build $(BUILD_DIR) -j$(JOBS)"
	@echo "  make clean            rm -rf $(BUILD_DIR)"
	@echo ""
	@echo "Test:"
	@echo "  make test             full ctest suite (22 tests)"
	@echo "  make bench            GEMM + attention + 7B Q4_0 inference benches"
	@echo "  make smoke            release_smoke.sh (REQUIRE_GPU=1)"
	@echo ""
	@echo "Examples:"
	@echo "  make hello            ./build/examples/hello_gemm"
	@echo "  make inspect FILE=x   ./build/examples/gguf_inspect <FILE>"
	@echo "  make decode           ./build/examples/decode_step"
	@echo "  make train            ./build/examples/training_step"
	@echo ""
	@echo "Quality gates:"
	@echo "  make check-version    scripts/check_version_consistency.sh"
	@echo "  make check-headers    scripts/check_public_headers.sh"
	@echo "  make check-exports    scripts/check_public_exports.sh"
	@echo "  make check-python     scripts/check_python_{ffi_surface,abi_layout,constants}.py"
	@echo "  make docs-check       no orphan docs, no broken intra-doc links"
	@echo "  make icc-audit        re-run ICC index + doc-coverage + shell-hardening"
	@echo ""
	@echo "Packaging:"
	@echo "  make install          cmake --install $(BUILD_DIR) --prefix $(INSTALL_PREFIX)"
	@echo "  make wheel            build a wheel + verify it imports"
	@echo ""
	@echo "Variables (override on command line):"
	@echo "  BUILD_DIR=$(BUILD_DIR)  BUILD_TYPE=$(BUILD_TYPE)  JOBS=$(JOBS)"
	@echo "  INSTALL_PREFIX=$(INSTALL_PREFIX)  WHEEL_DIR=$(WHEEL_DIR)  PYTHON=$(PYTHON)"

# --- Build ----------------------------------------------------------------

configure:
	cmake -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

build: configure
	cmake --build $(BUILD_DIR) -j$(JOBS)

clean:
	rm -rf $(BUILD_DIR)

# --- Test / bench / smoke -------------------------------------------------

test: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure

bench: build
	$(BUILD_DIR)/bench/bench_gemm
	$(BUILD_DIR)/bench/bench_attention
	$(BUILD_DIR)/bench/bench_inference_7b

smoke: build
	REQUIRE_GPU=1 scripts/release_smoke.sh

# --- Examples -------------------------------------------------------------

hello: build
	$(BUILD_DIR)/examples/hello_gemm

inspect: build
	@if [ -z "$(FILE)" ]; then \
	  echo "usage: make inspect FILE=path/to/model.gguf"; \
	  exit 1; \
	fi
	$(BUILD_DIR)/examples/gguf_inspect $(FILE)

decode: build
	$(BUILD_DIR)/examples/decode_step

train: build
	$(BUILD_DIR)/examples/training_step

examples: hello decode train

# --- Quality gates --------------------------------------------------------

check-version:
	scripts/check_version_consistency.sh

check-headers: build
	scripts/check_public_headers.sh

check-exports: build
	scripts/check_public_exports.sh

check-python: build
	$(PYTHON) scripts/check_python_ffi_surface.py
	$(PYTHON) scripts/check_python_abi_layout.py
	$(PYTHON) scripts/check_python_constants.py

docs-check:
	$(PYTHON) scripts/check_docs_links.py

icc-audit:
	@if [ -z "$$ICC_HOME" ]; then \
	  echo "Set ICC_HOME to the infinite_context_coder checkout to run this target."; \
	  exit 1; \
	fi
	$$ICC_HOME/bin/icc index --repo tensorcore
	$$ICC_HOME/bin/icc build-memory --repo tensorcore
	$$ICC_HOME/bin/icc unreferenced-docs --repo tensorcore
	$$ICC_HOME/bin/icc doc-coverage --repo tensorcore
	$$ICC_HOME/bin/icc audit-patterns --repo tensorcore --preset shell-hardening

# --- Packaging ------------------------------------------------------------

install: build
	cmake --install $(BUILD_DIR) --prefix $(INSTALL_PREFIX)

wheel: install
	@mkdir -p $(WHEEL_DIR)
	@if [ ! -d /tmp/tc-wheel-venv ]; then \
	  $(PYTHON) -m venv /tmp/tc-wheel-venv; \
	fi
	. /tmp/tc-wheel-venv/bin/activate && \
	  $(PYTHON) -m pip install --upgrade pip setuptools wheel >/dev/null && \
	  TENSORCORE_NATIVE_DIR=$(INSTALL_PREFIX)/lib $(PYTHON) -m pip wheel . \
	    --no-build-isolation -w $(WHEEL_DIR) && \
	  $(PYTHON) -m pip install $(WHEEL_DIR)/tensorcore_apple-*.whl \
	    --force-reinstall --no-deps && \
	  TENSORCORE_LIB= TC_METALLIB= $(PYTHON) -c \
	    'import tensorcore as tc; print(tc.version())'

# --- Composite ------------------------------------------------------------

all: build test
