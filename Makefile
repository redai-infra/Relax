.PHONY: help install test lint format clean docs docs-dev docs-build docs-preview check-registry check-train-image docker-train docker-dev

DOCKER ?= docker
DOCKERFILE ?= docker/Dockerfile
DOCKER_BUILD_PROGRESS ?= plain
DOCKER_BUILD_ARGS ?=
DO_PUSH ?= 1
IMAGE_REPOSITORY ?= relax
BUILD_DATE := $(shell date +%Y%m%d)
GIT_SHORT_HASH := $(shell git rev-parse --short=8 HEAD)

IMAGE_REGISTRY := $(patsubst %/,%,$(strip $(REGISTRY)))
DEFAULT_TRAIN_IMAGE := $(IMAGE_REGISTRY)/$(IMAGE_REPOSITORY):train-$(BUILD_DATE)-$(GIT_SHORT_HASH)
DEV_IMAGE := $(IMAGE_REGISTRY)/$(IMAGE_REPOSITORY):dev-$(BUILD_DATE)-$(GIT_SHORT_HASH)

ifeq ($(origin TRAIN_IMAGE),undefined)
TRAIN_IMAGE := $(DEFAULT_TRAIN_IMAGE)
BUILD_DEFAULT_TRAIN := 1
else
BUILD_DEFAULT_TRAIN := 0
endif

export HTTP_PROXY HTTPS_PROXY NO_PROXY

PROXY_BUILD_ARGS = --build-arg HTTP_PROXY --build-arg HTTPS_PROXY --build-arg NO_PROXY
BASE_IMAGE_BUILD_ARG = $(if $(strip $(BASE_IMAGE)),--build-arg BASE_IMAGE="$(BASE_IMAGE)")
IMAGE_INSPECT := $(DOCKER) $(if $(filter 0,$(DO_PUSH)),image,manifest) inspect
IMAGE_LOCATION := $(if $(filter 0,$(DO_PUSH)),local,remote)

help:
	@echo "Available commands:"
	@echo "  make install       - Install dependencies"
	@echo "  make test          - Run tests"
	@echo "  make lint          - Run linters"
	@echo "  make format        - Format code"
	@echo "  make clean         - Clean build artifacts"
	@echo "  make docs-dev      - Start documentation dev server"
	@echo "  make docs-build    - Build documentation"
	@echo "  make docs-preview  - Preview built documentation"
	@echo "  REGISTRY=... make docker-train - Build and push the Docker train stage"
	@echo "  REGISTRY=... make docker-dev   - Build and push the Docker development image"
	@echo "  REGISTRY=... TRAIN_IMAGE=... make docker-dev - Build dev from an existing train image"
	@echo "  Set DO_PUSH=0 before make to skip pushing Docker images"
	@echo "  Existing remote images are skipped; DO_PUSH=0 checks local images"

install:
	pip install -e .
	pip install -r requirements.txt

test:
	pytest tests/

lint:
	flake8 relax/
	mypy relax/

format: # develop ## Code format using pre-commit tools
	@which pre-commit 2>&1 > /dev/null || python -m pip install pre-commit==3.8.0
	@pre-commit run --all-files --show-diff-on-failure

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Documentation commands
docs-dev:
	cd docs && npm install && npm run docs:dev

docs-build:
	cd docs && npm install && npm run docs:build

docs-preview:
	cd docs && npm run docs:preview

docs-install:
	cd docs && npm install

check-registry:
	@test -n "$(IMAGE_REGISTRY)" || { echo "REGISTRY is required" >&2; exit 2; }

check-train-image:
	@test -n "$(strip $(TRAIN_IMAGE))" || { echo "TRAIN_IMAGE must not be empty" >&2; exit 2; }

docker-train: check-registry
	@echo "[docker] output train image: $(TRAIN_IMAGE)"
	@set -e; \
	if $(IMAGE_INSPECT) "$(TRAIN_IMAGE)" >/dev/null 2>&1; then \
		echo "[docker] skip existing $(IMAGE_LOCATION) train image: $(TRAIN_IMAGE)"; \
	else \
		$(DOCKER) build --progress=$(DOCKER_BUILD_PROGRESS) \
			-f $(DOCKERFILE) \
			--target train \
			-t "$(TRAIN_IMAGE)" \
			$(PROXY_BUILD_ARGS) $(BASE_IMAGE_BUILD_ARG) $(DOCKER_BUILD_ARGS) \
			.; \
		if [ "$(DO_PUSH)" != "0" ]; then $(DOCKER) push "$(TRAIN_IMAGE)"; fi; \
	fi

docker-dev: check-registry check-train-image
	@echo "[docker] input train image: $(TRAIN_IMAGE)"
	@echo "[docker] output dev image: $(DEV_IMAGE)"
	@set -e; \
	if $(IMAGE_INSPECT) "$(DEV_IMAGE)" >/dev/null 2>&1; then \
		echo "[docker] skip existing $(IMAGE_LOCATION) dev image: $(DEV_IMAGE)"; \
	else \
		if [ "$(BUILD_DEFAULT_TRAIN)" = "1" ]; then \
			$(MAKE) --no-print-directory docker-train TRAIN_IMAGE="$(TRAIN_IMAGE)"; \
		fi; \
		$(DOCKER) build --progress=$(DOCKER_BUILD_PROGRESS) \
			-f $(DOCKERFILE) \
			--target relax \
			-t "$(DEV_IMAGE)" \
			--build-arg TRAIN_IMAGE="$(TRAIN_IMAGE)" \
			$(PROXY_BUILD_ARGS) $(BASE_IMAGE_BUILD_ARG) $(DOCKER_BUILD_ARGS) \
			.; \
		if [ "$(DO_PUSH)" != "0" ]; then $(DOCKER) push "$(DEV_IMAGE)"; fi; \
	fi
