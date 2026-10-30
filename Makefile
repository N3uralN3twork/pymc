# Makefile for documentation and local development
#

# You can set these variables from the command line.
SPHINXBUILD   = sphinx-build
SOURCEDIR     = docs/source
BUILDDIR      = docs/build
PYTHON        ?= $(if $(wildcard .venv/Scripts/python.exe),.venv/Scripts/python.exe,$(if $(wildcard .venv/bin/python),.venv/bin/python,python))
PYTEST_ARGS   ?= -v
ASV_CONFIG    ?= benchmarks/asv.conf.json
ASV_BENCH     ?= bench_kernels
ASV_ARGS      ?=

rtd: export READTHEDOCS=true

# User-friendly check for sphinx-build
ifneq ($(filter html rtd,$(MAKECMDGOALS)),)
ifeq ($(shell which $(SPHINXBUILD) >/dev/null 2>&1; echo $$?), 1)
$(error The '$(SPHINXBUILD)' command was not found. Make sure you have Sphinx installed, then set the SPHINXBUILD environment variable to point to the full path of the '$(SPHINXBUILD)' executable. Alternatively you can add the directory with the executable to your PATH. If you don't have Sphinx installed, grab it from http://sphinx-doc.org/)
endif
endif

.PHONY: help clean html rtd view setup test rust-release bench bench-quick bench-machine

setup:
	@echo "Installing dependencies:"
	uv sync

test:
	@"$(PYTHON)" scripts/run_tests.py $(PYTEST_ARGS)

# Build and install the optimized extension into the selected Python environment.
rust-release:
	@"$(PYTHON)" scripts/build_rsmc.py

# Compare both implementation parameters in the same installed environment.
# Rebuild rsmc in release mode after changing Rust sources.
bench:
	@"$(PYTHON)" -m asv run --config "$(ASV_CONFIG)" --python=same --bench "$(ASV_BENCH)" --show-stderr --dry-run $(ASV_ARGS)

bench-quick:
	@"$(PYTHON)" -m asv run --config "$(ASV_CONFIG)" --python=same --bench "$(ASV_BENCH)" --show-stderr --dry-run --quick $(ASV_ARGS)

bench-machine:
	@"$(PYTHON)" -m asv machine --yes --config "$(ASV_CONFIG)"

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html       to make standalone HTML files"
	@echo "  rtd        to build the website without any cache"
	@echo "  clean      to clean cache and intermediate files"
	@echo "  view       to open the built html files"
	@echo "  test       to select one or more unit test areas interactively"
	@echo "             Optional: override PYTEST_ARGS to pass pytest options"
	@echo "  rust-release to build/rebuild and install the Rust extension in release mode"
	@echo "             Optional: override PYTHON to select the virtual environment"
	@echo "  bench      to compare Python and Rust kernel latency, memory, throughput and speedup"
	@echo "  bench-quick   to smoke-check both kernel implementations with ASV"
	@echo "  bench-machine to initialize ASV machine metadata using detected defaults"
	@echo "             Optional: override ASV_BENCH, ASV_ARGS, or PYTHON"

clean:
	rm -rf $(BUILDDIR)/*
	rm -rf $(SOURCEDIR)/api/generated
	rm -rf $(SOURCEDIR)/api/**/generated
	rm -rf $(SOURCEDIR)/api/**/classmethods
	rm -rf docs/jupyter_execute

html:
	$(SPHINXBUILD) $(SOURCEDIR) $(BUILDDIR) -b html
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)."

rtd: clean
	$(SPHINXBUILD) $(SOURCEDIR) $(BUILDDIR) -b html -E
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)."

view:
	python -m webbrowser $(BUILDDIR)/index.html
