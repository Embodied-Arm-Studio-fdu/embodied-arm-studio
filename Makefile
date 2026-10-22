NODE_DIR := $(CURDIR)/.tools/node-v20.19.4-linux-x64/bin
PYTHON := /usr/bin/python3

.PHONY: test install-web dev stack-up stack-status stack-logs stack-down stack-restart mock-demo gpu-smoke build-web e2e

test:
	env -u DATABASE_URL -u TEST_DATABASE_URL -u POSTGRES_TEST_OPT_IN \
		RUN_REPOSITORY_MODE=memory PYTHONPATH=. \
		$(PYTHON) -m unittest discover -s tests -v

install-web:
	PATH=$(NODE_DIR):$$PATH npm --prefix apps/web install --no-audit --no-fund

build-web:
	PATH=$(NODE_DIR):$$PATH npm --prefix apps/web run build

e2e:
	PATH=$(NODE_DIR):$$PATH npm --prefix apps/web run e2e

dev:
	./scripts/dev.sh

stack-up:
	./scripts/studio.sh up

stack-status:
	./scripts/studio.sh status

stack-logs:
	./scripts/studio.sh logs

stack-down:
	./scripts/studio.sh down

stack-restart:
	./scripts/studio.sh restart

mock-demo:
	PYTHONPATH=. $(PYTHON) -m unittest tests.test_mock_adapter tests.test_api tests.test_agent_workspace -v

gpu-smoke:
	PYTHONPATH=. $(PYTHON) scripts/gpu_smoke.py
