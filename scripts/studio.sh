#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ENV_FILE="$ROOT_DIR/.env"
PROJECT_NAME=${COMPOSE_PROJECT_NAME:-$(basename "$ROOT_DIR")}

compose() {
    if docker compose version >/dev/null 2>&1; then
        docker compose --project-directory "$ROOT_DIR" --env-file "$ENV_FILE" "$@"
    elif command -v docker-compose >/dev/null 2>&1; then
        docker-compose --project-directory "$ROOT_DIR" --env-file "$ENV_FILE" "$@"
    else
        echo "Docker Compose is not installed." >&2
        exit 1
    fi
}

container_for_service() {
    docker ps -aq \
        --filter "label=com.docker.compose.project=$PROJECT_NAME" \
        --filter "label=com.docker.compose.service=$1" | head -n 1
}

container_environment_value() {
    local container=$1
    local name=$2
    docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$container" |
        sed -n "s/^${name}=//p" | head -n 1
}

adopt_existing_environment() {
    if [[ -e "$ENV_FILE" ]]; then
        echo "$ENV_FILE already exists; refusing to replace it." >&2
        exit 1
    fi

    local postgres_container
    local api_container
    postgres_container=$(container_for_service postgres)
    api_container=$(container_for_service api)
    if [[ -z "$postgres_container" || -z "$api_container" ]]; then
        echo "Existing Compose postgres and api containers were not both found." >&2
        exit 1
    fi

    local postgres_user
    local postgres_password
    local postgres_database
    local database_url
    local jwt_secret
    local jwt_issuer
    local jwt_audience
    local openai_api_key
    local openai_model
    local openai_base_url
    local deepseek_api_key
    local deepseek_model
    local anthropic_api_key
    local anthropic_model
    postgres_user=$(container_environment_value "$postgres_container" POSTGRES_USER)
    postgres_password=$(container_environment_value "$postgres_container" POSTGRES_PASSWORD)
    postgres_database=$(container_environment_value "$postgres_container" POSTGRES_DB)
    database_url=$(container_environment_value "$api_container" DATABASE_URL)
    jwt_secret=$(container_environment_value "$api_container" USER_JWT_SECRET)
    jwt_issuer=$(container_environment_value "$api_container" USER_JWT_ISSUER)
    jwt_audience=$(container_environment_value "$api_container" USER_JWT_AUDIENCE)
    openai_api_key=$(container_environment_value "$api_container" OPENAI_API_KEY)
    openai_model=$(container_environment_value "$api_container" OPENAI_MODEL)
    openai_base_url=$(container_environment_value "$api_container" OPENAI_BASE_URL)
    deepseek_api_key=$(container_environment_value "$api_container" DEEPSEEK_API_KEY)
    deepseek_model=$(container_environment_value "$api_container" DEEPSEEK_MODEL)
    anthropic_api_key=$(container_environment_value "$api_container" ANTHROPIC_API_KEY)
    anthropic_model=$(container_environment_value "$api_container" ANTHROPIC_MODEL)

    if [[ -z "$postgres_user" || -z "$postgres_password" || -z "$postgres_database" || -z "$database_url" || -z "$jwt_secret" ]]; then
        echo "Existing containers do not contain the required database and JWT configuration." >&2
        exit 1
    fi

    umask 077
    printf '%s\n' \
        '# Adopted from the existing local Compose runtime. Do not commit or regenerate.' \
        "POSTGRES_USER=$postgres_user" \
        "POSTGRES_PASSWORD=$postgres_password" \
        "POSTGRES_DB=$postgres_database" \
        "DATABASE_URL=$database_url" \
        "USER_JWT_SECRET=$jwt_secret" \
        "USER_JWT_ISSUER=${jwt_issuer:-embodied-arm-studio}" \
        "USER_JWT_AUDIENCE=${jwt_audience:-embodied-arm-studio-api}" \
        "OPENAI_API_KEY=$openai_api_key" \
        "OPENAI_MODEL=$openai_model" \
        "OPENAI_BASE_URL=${openai_base_url:-https://api.openai.com/v1}" \
        "DEEPSEEK_API_KEY=$deepseek_api_key" \
        "DEEPSEEK_MODEL=$deepseek_model" \
        "ANTHROPIC_API_KEY=$anthropic_api_key" \
        "ANTHROPIC_MODEL=$anthropic_model" >"$ENV_FILE"
    chmod 600 "$ENV_FILE"
    echo "Adopted existing database and JWT configuration into $ENV_FILE"
}

initialize_environment() {
    if [[ -f "$ENV_FILE" ]]; then
        return
    fi

    if [[ -n "$(container_for_service postgres)" ]]; then
        echo "An existing PostgreSQL container was found but $ENV_FILE is missing." >&2
        echo "Run '$0 adopt' before starting the stack so existing credentials are preserved." >&2
        exit 1
    fi

    umask 077
    local database_password
    local jwt_secret
    database_password=$(openssl rand -hex 24)
    jwt_secret=$(openssl rand -hex 48)
    printf '%s\n' \
        '# Private local runtime configuration. Do not commit or regenerate on restart.' \
        'POSTGRES_USER=studio' \
        "POSTGRES_PASSWORD=$database_password" \
        'POSTGRES_DB=embodied_arm_studio' \
        "DATABASE_URL=postgresql+psycopg://studio:$database_password@postgres:5432/embodied_arm_studio" \
        "USER_JWT_SECRET=$jwt_secret" \
        'USER_JWT_ISSUER=embodied-arm-studio' \
        'USER_JWT_AUDIENCE=embodied-arm-studio-api' \
        'OPENAI_API_KEY=' \
        'OPENAI_MODEL=' \
        'OPENAI_BASE_URL=https://api.openai.com/v1' \
        'DEEPSEEK_API_KEY=' \
        'DEEPSEEK_MODEL=' \
        'ANTHROPIC_API_KEY=' \
        'ANTHROPIC_MODEL=' >"$ENV_FILE"
    chmod 600 "$ENV_FILE"
    echo "Created private runtime configuration: $ENV_FILE"
}

wait_until_healthy() {
    local attempts=60
    while (( attempts > 0 )); do
        if curl --fail --silent --show-error http://127.0.0.1:5173/api/health >/dev/null 2>&1; then
            echo "Studio is ready at http://127.0.0.1:5173"
            return
        fi
        attempts=$((attempts - 1))
        sleep 2
    done
    compose ps >&2
    echo "Studio did not become healthy within 120 seconds. Run '$0 logs' for details." >&2
    exit 1
}

require_docker() {
    if ! docker info >/dev/null 2>&1; then
        echo "Docker daemon is unavailable. Start Docker, then rerun this command." >&2
        exit 1
    fi
}

usage() {
    echo "Usage: $0 {adopt|up|status|logs|down|restart}"
}

command=${1:-status}
case "$command" in
    adopt)
        require_docker
        adopt_existing_environment
        ;;
    up)
        require_docker
        initialize_environment
        compose up --build -d
        wait_until_healthy
        compose ps
        ;;
    status)
        require_docker
        initialize_environment
        compose ps
        ;;
    logs)
        require_docker
        initialize_environment
        compose logs --tail=200 -f
        ;;
    down)
        require_docker
        initialize_environment
        compose down
        ;;
    restart)
        require_docker
        initialize_environment
        compose up --build -d --force-recreate
        wait_until_healthy
        compose ps
        ;;
    *)
        usage >&2
        exit 2
        ;;
esac