# Compile both requirements files (after editing .in files)
deps-compile:
	@echo "Compiling requirements.txt from requirements.in..."
	@. ./$(VENV)/bin/activate && pip-compile requirements.in --resolver=backtracking
	@echo "Compiling requirements-dev.txt from requirements-dev.in..."
	@. ./$(VENV)/bin/activate && pip-compile requirements-dev.in --resolver=backtracking
	@echo "✅ Dependencies compiled successfully!"

# Install all dependencies (production + dev)
deps-install:
	@echo "Installing all dependencies..."
	@. ./$(VENV)/bin/activate && pip install -r requirements.txt -r requirements-dev.txt
	@echo "✅ Dependencies installed successfully!"

# Upgrade all dependencies to latest compatible versions
deps-upgrade:
	@echo "Upgrading all dependencies..."
	@. ./$(VENV)/bin/activate && pip-compile --upgrade requirements.in --resolver=backtracking
	@. ./$(VENV)/bin/activate && pip-compile --upgrade requirements-dev.in --resolver=backtracking
	@echo "✅ Dependencies upgraded! Review changes before committing."

# Upgrade a specific package
deps-upgrade-package:
	@if [ -z "$(PKG)" ]; then echo "Usage: make deps-upgrade-package PKG=package-name"; exit 1; fi
	@echo "Upgrading $(PKG)..."
	@. ./$(VENV)/bin/activate && pip-compile --upgrade-package $(PKG) requirements.in --resolver=backtracking
	@. ./$(VENV)/bin/activate && pip-compile --upgrade-package $(PKG) requirements-dev.in --resolver=backtracking
	@echo "✅ $(PKG) upgraded!"

# Sync venv to exactly match requirements files (removes unused packages)
deps-sync:
	@echo "Syncing environment to requirements files..."
	@. ./$(VENV)/bin/activate && pip-sync requirements.txt requirements-dev.txt
	@echo "✅ Environment synced!"

# Check for dependency conflicts and vulnerabilities
deps-check:
	@echo "Checking for dependency conflicts..."
	@. ./$(VENV)/bin/activate && pip check
	@echo "Checking for security vulnerabilities..."
	@. ./$(VENV)/bin/activate && pip-audit
	@echo "✅ Dependency check complete!"

# Show dependency tree
deps-tree:
	@echo "Dependency tree:"
	@. ./$(VENV)/bin/activate && pipdeptree

# Show outdated packages
deps-outdated:
	@echo "Checking for outdated packages..."
	@. ./$(VENV)/bin/activate && pip list --outdated
