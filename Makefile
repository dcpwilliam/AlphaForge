.PHONY: up down venv install fetch-real fetch-real-100 load update-incremental \
	schedule-install schedule-uninstall notebook web test-conn

up:
	docker compose up -d

down:
	docker compose down

venv:
	python3 -m venv .venv

install:
	. .venv/bin/activate && pip install -U pip setuptools wheel && pip install .[dev]

fetch-real:
	. .venv/bin/activate && PYTHONPATH=src python scripts/fetch_real_market_data.py --pool-size 50

fetch-real-100:
	. .venv/bin/activate && PYTHONPATH=src python scripts/fetch_real_market_data.py --pool-size 100

load:
	. .venv/bin/activate && PYTHONPATH=src python scripts/load_to_clickhouse.py

update-incremental:
	. .venv/bin/activate && PYTHONPATH=src python scripts/incremental_update_clickhouse.py --pool-size 100

schedule-install:
	chmod +x scripts/run_incremental_update.sh scripts/install_launchd_job.sh
	./scripts/install_launchd_job.sh --hour 8 --minute 10 --pool-size 100

schedule-uninstall:
	./scripts/install_launchd_job.sh --uninstall

notebook:
	. .venv/bin/activate && PYTHONPATH=src jupyter lab

web:
	. .venv/bin/activate && PYTHONPATH=src streamlit run web/app.py

test-conn:
	. .venv/bin/activate && PYTHONPATH=src python -c "from quant_research.db import query_df; print(query_df('select version() as v'))"
