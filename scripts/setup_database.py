#!/usr/bin/env python3
"""
Database Setup Script for Grounded Memory System

This script helps set up the PostgreSQL database for the Grounded Memory System.
It can:
1. Create the database if it doesn't exist
2. Initialize the schema
3. Verify the connection

Usage:
    python scripts/setup_database.py --create-db
    python scripts/setup_database.py --init-schema
    python scripts/setup_database.py --verify
    python scripts/setup_database.py --all

Environment Variables (from .env):
    POSTGRES_HOST     - PostgreSQL host (default: localhost)
    POSTGRES_PORT     - PostgreSQL port (default: 5432)
    POSTGRES_DB       - Database name (default: grounded_memory)
    POSTGRES_USER     - PostgreSQL user (default: postgres)
    POSTGRES_PASSWORD - PostgreSQL password
"""

# Load environment variables from .env file
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv not available, use system env vars

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def get_config():
    """Get database configuration from environment."""
    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "database": os.getenv("POSTGRES_DB", "grounded_memory"),
        "user": os.getenv("POSTGRES_USER", "postgres"),
        "password": os.getenv("POSTGRES_PASSWORD", ""),
    }


def create_database():
    """Create the grounded_memory database if it doesn't exist."""
    try:
        import psycopg2
        from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
    except ImportError:
        print("❌ psycopg2 not installed. Install with: pip install psycopg2-binary")
        return False

    config = get_config()
    db_name = config["database"]

    print(f"Creating database '{db_name}'...")

    try:
        # Connect to default 'postgres' database to create new database
        conn = psycopg2.connect(
            host=config["host"],
            port=config["port"],
            database="postgres",
            user=config["user"],
            password=config["password"],
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

        cursor = conn.cursor()

        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))

        if cursor.fetchone():
            print(f"✓ Database '{db_name}' already exists")
        else:
            cursor.execute(f'CREATE DATABASE "{db_name}"')
            print(f"✓ Database '{db_name}' created successfully")

        cursor.close()
        conn.close()
        return True

    except Exception as e:
        print(f"❌ Failed to create database: {e}")
        return False


async def init_schema():
    """Initialize the database schema."""
    try:
        from grounded_memory.core.postgres_store import PostgresConfig, PostgresStore
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure to install with: pip install -e '.[postgres]'")
        return False

    config = PostgresConfig.from_env()

    print(f"Initializing schema on {config.host}:{config.port}/{config.database}...")

    try:
        store = PostgresStore(config)
        await store.initialize(create_schema=True)

        # Verify by getting statistics
        await store.get_statistics()
        print("✓ Schema initialized successfully")
        print("  Tables created: entities, validated_facts, interactions, rejection_records")

        await store.close()
        return True

    except Exception as e:
        print(f"❌ Failed to initialize schema: {e}")
        return False


async def verify_connection():
    """Verify database connection and schema."""
    try:
        from grounded_memory.core.postgres_store import PostgresConfig, PostgresStore
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

    config = PostgresConfig.from_env()

    print(f"Verifying connection to {config.host}:{config.port}/{config.database}...")

    try:
        store = PostgresStore(config)
        await store.initialize(create_schema=False)

        # Get statistics
        stats = await store.get_statistics()

        print("✓ Connection successful!")
        print("\nDatabase Statistics:")
        print(f"  Entities: {stats.get('total_entities', 0)}")
        print(f"  Active Facts: {stats.get('active_facts', 0)}")
        print(f"  Superseded Facts: {stats.get('superseded_facts', 0)}")
        print(f"  Interactions: {stats.get('total_interactions', 0)}")
        print(f"  Rejections: {stats.get('total_rejections', 0)}")

        if stats.get("entities_by_type"):
            print("\n  Entities by Type:")
            for etype, count in stats["entities_by_type"].items():
                print(f"    - {etype}: {count}")

        await store.close()
        return True

    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False


def print_docker_compose():
    """Print Docker Compose configuration for PostgreSQL."""
    compose = """
# docker-compose.yml for Grounded Memory PostgreSQL

version: '3.8'

services:
  postgres:
    image: postgres:16-alpine
    container_name: grounded_memory_db
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: your_secure_password_here
      POSTGRES_DB: grounded_memory
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  postgres_data:
"""
    print(compose)


def print_env_template():
    """Print environment variable template."""
    template = """
# .env file for Grounded Memory System

# PostgreSQL Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=grounded_memory
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_secure_password_here

# LLM Configuration (OpenRouter)
LLM_PROVIDER=openrouter
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Optional: Override LLM settings
# LLM_MODEL=anthropic/claude-3.5-sonnet
# LLM_TEMPERATURE=0.1
# LLM_MAX_TOKENS=2048
"""
    print(template)


def main():
    parser = argparse.ArgumentParser(description="Database setup script for Grounded Memory System")
    parser.add_argument(
        "--create-db", action="store_true", help="Create the database if it doesn't exist"
    )
    parser.add_argument("--init-schema", action="store_true", help="Initialize the database schema")
    parser.add_argument(
        "--verify", action="store_true", help="Verify database connection and schema"
    )
    parser.add_argument(
        "--docker-compose", action="store_true", help="Print Docker Compose configuration"
    )
    parser.add_argument("--env-template", action="store_true", help="Print .env file template")
    parser.add_argument(
        "--all", action="store_true", help="Run all setup steps (create-db + init-schema + verify)"
    )

    args = parser.parse_args()

    # If no arguments, show help
    if not any(vars(args).values()):
        parser.print_help()
        return

    print("=" * 60)
    print("Grounded Memory Database Setup")
    print("=" * 60)

    success = True

    if args.docker_compose:
        print("\nDocker Compose Configuration:")
        print("-" * 60)
        print_docker_compose()
        return

    if args.env_template:
        print("\nEnvironment Variables Template:")
        print("-" * 60)
        print_env_template()
        return

    if args.create_db or args.all:
        print("\n[1/3] Creating Database...")
        print("-" * 60)
        success = create_database() and success

    if args.init_schema or args.all:
        print("\n[2/3] Initializing Schema...")
        print("-" * 60)
        success = asyncio.run(init_schema()) and success

    if args.verify or args.all:
        print("\n[3/3] Verifying Connection...")
        print("-" * 60)
        success = asyncio.run(verify_connection()) and success

    print("\n" + "=" * 60)
    if success:
        print("✓ Setup completed successfully!")
    else:
        print("❌ Some steps failed. Check the errors above.")
    print("=" * 60)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
