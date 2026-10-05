# Tasks for the react_loop project.
# Run `make` or `make help` to see what is available.

UV ?= uv
PY := $(UV) run python
PKG := react_loop
EXTRAS := --extra openai --extra anthropic
DEMO_QUESTION ?= Express shipping costs $$15 per order. How much is it for 3 items, and what is the refund window?

.DEFAULT_GOAL := help
.PHONY: help install sync chat demo stream run stream-run ask test lint fmt check clean env

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Install every dependency, including the provider extras
	$(UV) sync $(EXTRAS)

sync: ## Install the base dependencies only (offline demo)
	$(UV) sync

env: ## Create .env from the template if it is missing
	@test -f .env || (cp .env.example .env && echo "created .env from .env.example")

chat: ## Start an interactive session that remembers the conversation
	$(UV) run react-loop

demo: ## Run the offline demo. No API key needed
	$(PY) -m $(PKG) --demo

stream: ## Run the offline demo with token streaming
	$(PY) -m $(PKG) --demo --stream

run: ## Run a question, e.g. make run Q="What is 21 * 2?"
	$(PY) -m $(PKG) "$(or $(Q),$(DEMO_QUESTION))"

stream-run: ## Run a question with streaming, e.g. make stream-run Q="Why?"
	$(PY) -m $(PKG) --stream "$(or $(Q),$(DEMO_QUESTION))"

ask: ## Alias for run
	$(MAKE) run Q="$(Q)"

test: ## Run the test suite
	$(UV) run pytest

lint: ## Check lint rules
	$(UV) run ruff check .

fmt: ## Apply safe lint fixes
	$(UV) run ruff check . --fix

check: lint test ## Run lint and tests

clean: ## Remove caches and build artefacts
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find src tests -name __pycache__ -type d -exec rm -rf {} +
