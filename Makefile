.PHONY: all debug build wasm clean fmt install test

all: clean fmt build install wasm

build:
	cargo build --release

debug:
	cargo build --debug

install:
ifeq ($(shell uname),Darwin)
	rustup target add $$(python3 -c 'import platform; print(platform.machine())')-apple-darwin 2>/dev/null || true
endif
	uv pip install -e ".[all]"

WASM_BINDGEN_VERSION := $(shell sed -n 's/^wasm-bindgen.*"=\([0-9.]*\)".*/\1/p' Cargo.toml)

wasm:
	rustup target add wasm32-unknown-unknown 2>/dev/null || true
	@cargo install --list | grep -q "^wasm-bindgen-cli v$(WASM_BINDGEN_VERSION):" || \
	  cargo install wasm-bindgen-cli --version $(WASM_BINDGEN_VERSION) --force
	cargo build -p rlcade --no-default-features --features wasm --target wasm32-unknown-unknown --release
	mkdir -p $(CURDIR)/pkg
	wasm-bindgen target/wasm32-unknown-unknown/release/nes.wasm --out-dir $(CURDIR)/pkg --target web

clean:
	cargo clean
	rm -rf pkg/

fmt:
	cargo fmt --all
	cargo clippy -p rlcade -- -D warnings
	black rlcade/ tests/ bench/
	shfmt -w -i 2 examples/

test:
	python -m pytest tests/ -v
