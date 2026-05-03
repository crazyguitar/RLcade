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
	maturin develop --release
	pip install -e ".[all]"

wasm:
	rustup target add wasm32-unknown-unknown 2>/dev/null || true
	cargo install wasm-bindgen-cli 2>/dev/null || true
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
