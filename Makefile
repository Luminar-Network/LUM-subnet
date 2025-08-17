# Luminar Subnet Database Management
# Copyright © 2025 Luminar Network

.PHONY: help db-setup db-start db-stop db-migrate db-reset db-backup db-restore test-db

# Default environment
ENV ?= local

help: ## Show this help message
	@echo "Luminar Subnet Database Management"
	@echo "=================================="
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Database Setup
db-setup: ## Set up local database with Docker
	@echo "Setting up Luminar database..."
	docker-compose up -d luminar-db
	@echo "Waiting for database to be ready..."
	sleep 10
	@echo "Running migrations..."
	python scripts/setup_database.py setup --env=$(ENV)
	@echo "Database setup complete!"

db-start: ## Start database containers
	docker-compose up -d luminar-db

db-stop: ## Stop database containers
	docker-compose down

db-migrate: ## Run database migrations
	python migrations/manager.py migrate

db-reset: ## Reset database (WARNING: Destroys all data)
	@echo "WARNING: This will destroy all data in the database!"
	@read -p "Are you sure? (y/N): " confirm && [ "$$confirm" = "y" ] || exit 1
	docker-compose down -v
	docker-compose up -d luminar-db
	sleep 10
	python scripts/setup_database.py setup --env=$(ENV)

db-backup: ## Create database backup
	@mkdir -p backups
	python scripts/setup_database.py backup --env=$(ENV)

db-test-setup: ## Set up test database
	docker-compose --profile testing up -d luminar-test-db
	sleep 5
	LUMINAR_ENV=testing python scripts/setup_database.py setup

# Migration Management
migrate-create: ## Create new migration (usage: make migrate-create NAME=migration_name)
	@if [ -z "$(NAME)" ]; then echo "Usage: make migrate-create NAME=migration_name"; exit 1; fi
	python migrations/manager.py create $(NAME)

migrate-status: ## Show migration status
	python migrations/manager.py status

# Development
dev-setup: ## Complete development setup
	@echo "Setting up Luminar development environment..."
	cp .env.local .env
	make db-setup
	pip install -r requirements.txt
	@echo "Development setup complete!"

# Testing
test: ## Run tests with test database
	make db-test-setup
	python -m pytest tests/ -v
	docker-compose --profile testing down

# Monitoring
monitoring-start: ## Start monitoring stack (Grafana + Prometheus)
	docker-compose --profile monitoring up -d

monitoring-stop: ## Stop monitoring stack
	docker-compose --profile monitoring down

# Cache
cache-start: ## Start Redis cache
	docker-compose --profile cache up -d luminar-redis

cache-stop: ## Stop Redis cache
	docker-compose --profile cache down

# Production helpers
prod-deploy: ## Deploy to production (use with caution)
	@echo "Deploying to production..."
	@echo "Make sure you have set LUMINAR_ENV=production"
	@if [ "$(ENV)" != "production" ]; then echo "ERROR: Must set ENV=production"; exit 1; fi
	python scripts/setup_database.py migrate --env=production

# Status and Health
db-status: ## Check database status
	docker-compose ps luminar-db
	python -c "import asyncio; from database.access import health_check; print('Healthy' if asyncio.run(health_check()) else 'Unhealthy')"

db-stats: ## Show database statistics
	python -c "import asyncio; from database.access import db; asyncio.run(db.initialize()); stats = asyncio.run(db.get_database_stats()); print(f'Reports: {stats.total_reports}, Events: {stats.processed_events}, Miners: {stats.active_miners}')"

# Cleanup
clean: ## Clean up all Docker resources
	docker-compose down -v --remove-orphans
	docker system prune -f

# Quick commands
quick-start: db-start ## Quick start database
	@echo "Database started on localhost:5432"
	@echo "Default credentials: luminar_dev / dev_password"

logs: ## Show database logs
	docker-compose logs -f luminar-db
