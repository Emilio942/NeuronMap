"""
CLI Commands for NeuronMap Analysis Zoo

Provides command-line interface for interacting with the Analysis Zoo:
- Authentication and login
- Pushing artifacts to the zoo
- Pulling artifacts from the zoo
- Searching for artifacts

Based on aufgabenliste_b.md Tasks C1-C4
"""

import os
import json
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import getpass
import keyring
import requests
from datetime import datetime

import click
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm

from ..zoo import (
    ArtifactSchema,
    ArtifactType,
    LicenseType,
    AuthorInfo,
    ArtifactManager,
    create_sae_artifact_template,
    create_circuit_artifact_template
)

console = Console()

# Configuration
ZOO_API_URL = os.getenv("ZOO_API_URL", "http://localhost:8001")
ZOO_KEYRING_SERVICE = "neuronmap-zoo"
ZOO_KEYRING_USERNAME = "api_token"
ZOO_LOCAL_CACHE = Path.home() / ".neuronmap" / "zoo_cache"


class ZooError(Exception):
    """Base exception for Zoo CLI operations."""
    pass


class ZooClient:
    """Client for interacting with the Analysis Zoo API."""

    def __init__(self, api_url: str = ZOO_API_URL):
        self.api_url = api_url.rstrip('/')
        self.session = requests.Session()

        # Load authentication token
        self.token = self._load_token()
        if self.token:
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})

    def _load_token(self) -> Optional[str]:
        """Load authentication token from keyring."""
        try:
            return keyring.get_password(ZOO_KEYRING_SERVICE, ZOO_KEYRING_USERNAME)
        except Exception:
            return None

    def _save_token(self, token: str):
        """Save authentication token to keyring."""
        try:
            keyring.set_password(ZOO_KEYRING_SERVICE, ZOO_KEYRING_USERNAME, token)
        except Exception as e:
            console.print(
                f"[yellow]Warning: Could not save token to keyring: {e}[/yellow]")

    def _delete_token(self):
        """Delete authentication token from keyring."""
        try:
            keyring.delete_password(ZOO_KEYRING_SERVICE, ZOO_KEYRING_USERNAME)
        except Exception:
            pass

    def login(self, token: str):
        """Login with API token."""
        # Test the token
        test_response = requests.get(
            f"{self.api_url}/artifacts",
            headers={"Authorization": f"Bearer {token}"},
            params={"limit": 1}
        )

        if test_response.status_code != 200:
            raise ZooError(f"Invalid token or API error: {test_response.status_code}")

        # Save token
        self._save_token(token)
        self.token = token
        self.session.headers.update({"Authorization": f"Bearer {token}"})

        console.print("[green]✓ Successfully logged in to Analysis Zoo[/green]")

    def logout(self):
        """Logout and clear stored credentials."""
        self._delete_token()
        self.token = None
        if "Authorization" in self.session.headers:
            del self.session.headers["Authorization"]

        console.print("[green]✓ Successfully logged out[/green]")

    def search_artifacts(
        self,
        query: str = None,
        artifact_type: str = None,
        tags: List[str] = None,
        model_name: str = None,
        limit: int = 50
    ) -> Dict[str, Any]:
        """Search for artifacts."""
        params = {"limit": limit}

        if query:
            params["query"] = query
        if artifact_type:
            params["artifact_type"] = artifact_type
        if tags:
            params["tags"] = ",".join(tags)
        if model_name:
            params["model_name"] = model_name

        if query:
            response = self.session.get(
                f"{self.api_url}/artifacts/search", params=params)
        else:
            response = self.session.get(f"{self.api_url}/artifacts", params=params)

        if response.status_code != 200:
            raise ZooError(f"Search failed: {response.status_code} - {response.text}")

        return response.json()

    def get_artifact(self, artifact_id: str) -> Dict[str, Any]:
        """Get specific artifact by ID."""
        response = self.session.get(f"{self.api_url}/artifacts/{artifact_id}")

        if response.status_code == 404:
            raise ZooError(f"Artifact {artifact_id} not found")
        elif response.status_code != 200:
            raise ZooError(f"Failed to get artifact: {
                           response.status_code} - {response.text}")

        return response.json()

    def download_artifact(self, artifact_id: str, target_path: Path) -> Path:
        """Download artifact to local path."""
        response = self.session.get(f"{self.api_url}/artifacts/{artifact_id}/download")

        if response.status_code != 200:
            raise ZooError(f"Download failed: {response.status_code} - {response.text}")

        # For now, just save the JSON manifest
        # TODO: Handle ZIP file downloads
        target_path.mkdir(parents=True, exist_ok=True)
        manifest_path = target_path / "artifact.json"

        with open(manifest_path, 'wb') as f:
            f.write(response.content)

        return target_path


# CLI commands

@click.group()
def zoo():
    """Analysis Zoo commands for artifact management."""
    pass


@zoo.command()
@click.option('--token', help='API token for authentication')
def login(token: str):
    """Login to the Analysis Zoo."""
    try:
        if not token:
            token = getpass.getpass("Enter your API token: ")

        client = ZooClient()
        client.login(token)

    except Exception as e:
        console.print(f"[red]Login failed: {e}[/red]")
        raise click.Abort()


@zoo.command()
def logout():
    """Logout from the Analysis Zoo."""
    try:
        client = ZooClient()
        client.logout()

    except Exception as e:
        console.print(f"[red]Logout failed: {e}[/red]")
        raise click.Abort()


def _get_artifact_details_from_prompts(name: str, artifact_type: str, description: str, author_name: str, author_email: str) -> Tuple[str, str, str, AuthorInfo]:
    if not name:
        name = Prompt.ask("Artifact name")

    if not artifact_type:
        type_choices = [
            'sae_model',
            'circuit',
            'intervention_config',
            'analysis_result',
            'dataset',
            'visualization']
        artifact_type = Prompt.ask("Artifact type", choices=type_choices)

    if not description:
        description = Prompt.ask("Brief description")

    if not author_name:
        author_name = Prompt.ask("Author name")

    author = AuthorInfo(
        name=author_name,
        email=author_email
    )
    return name, artifact_type, description, author


def _create_artifact_template(artifact_type: str, name: str, version: str, description: str, author: AuthorInfo, license_type: str, tag_list: List[str], model_name: Optional[str]) -> ArtifactSchema:
    if artifact_type == 'sae_model':
        if not model_name:
            model_name = Prompt.ask("Model name (e.g., gpt2, llama-2-7b)")
        artifact = create_sae_artifact_template(
            name=name,
            model_name=model_name,
            layer=0,  # TODO: make configurable
            dict_size=16384,  # TODO: make configurable
            authors=[author]
        )
    elif artifact_type == 'circuit':
        if not model_name:
            model_name = Prompt.ask("Model name")
        circuit_type = Prompt.ask("Circuit type (e.g., induction, copying)")
        artifact = create_circuit_artifact_template(
            name=name,
            model_name=model_name,
            circuit_type=circuit_type,
            authors=[author]
        )
    else:
        artifact = ArtifactSchema(
            name=name,
            version=version,
            artifact_type=ArtifactType(artifact_type),
            description=description,
            authors=[author],
            license=LicenseType(license_type),
            tags=tag_list,
            model_compatibility=[],
            files=[],
            total_size_bytes=0
        )
    artifact.version = version
    artifact.description = description
    artifact.license = LicenseType(license_type)
    artifact.tags = tag_list
    return artifact


@zoo.command()
@click.argument('path', type=click.Path(exists=True, path_type=Path))
@click.option('--name', help='Artifact name')
@click.option('--version', default='1.0.0', help='Artifact version')
@click.option('--type',
              'artifact_type',
              type=click.Choice(['sae_model',
                                 'circuit',
                                 'intervention_config',
                                 'analysis_result',
                                 'dataset',
                                 'visualization']),
              help='Artifact type')
@click.option('--description', help='Brief description')
@click.option('--license',
              'license_type',
              type=click.Choice(['MIT',
                                 'Apache-2.0',
                                 'CC-BY-4.0',
                                 'CC-BY-SA-4.0',
                                 'GPL-3.0',
                                 'Proprietary',
                                 'Custom']),
              default='MIT',
              help='License type')
@click.option('--tags', help='Comma-separated tags')
@click.option('--model-name', help='Compatible model name')
@click.option('--author-name', help='Author name')
@click.option('--author-email', help='Author email')
@click.option('--dry-run', is_flag=True, help='Validate but do not upload')
def push(
    path: Path,
    name: str,
    version: str,
    artifact_type: str,
    description: str,
    license_type: str,
    tags: str,
    model_name: str,
    author_name: str,
    author_email: str,
    dry_run: bool
):
    """Push an artifact to the Analysis Zoo."""
    try:
        name, artifact_type, description, author = _get_artifact_details_from_prompts(name, artifact_type, description, author_name, author_email)
        tag_list = [tag.strip() for tag in tags.split(",")] if tags else []

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            artifact_manager = ArtifactManager(temp_path)

            artifact = _create_artifact_template(artifact_type, name, version, description, author, license_type, tag_list, model_name)

            console.print("[cyan]Preparing artifact...[/cyan]")
            artifact = artifact_manager.prepare_artifact(path, artifact)

            console.print("[cyan]Validating artifact...[/cyan]")
            validated_artifact = artifact_manager.validate_artifact(path)

            console.print("\n[bold]Artifact Summary:[/bold]")
            console.print(f"Name: {validated_artifact.name}")
            console.print(f"Version: {validated_artifact.version}")
            console.print(f"Type: {validated_artifact.artifact_type.value}")
            console.print(f"Description: {validated_artifact.description}")
            console.print(f"License: {validated_artifact.license.value}")
            console.print(f"Files: {len(validated_artifact.files)}")
            console.print(f"Size: {validated_artifact.total_size_bytes:,} bytes")

            if dry_run:
                console.print("\n[green]✓ Dry run completed successfully[/green]")
                return

            if not Confirm.ask("Upload this artifact?"):
                console.print("[yellow]Upload cancelled[/yellow]")
                return

            console.print("[red]Note: Actual upload not yet implemented[/red]")
            console.print("[yellow]Artifact validated and ready for upload[/yellow]")

    except Exception as e:
        console.print(f"[red]Push failed: {e}[/red]")
        raise click.Abort()


@zoo.command()
@click.argument('artifact_id')
@click.option('--target', type=click.Path(path_type=Path), help='Target directory')
@click.option('--force', is_flag=True, help='Overwrite existing files')
def pull(artifact_id: str, target: Path, force: bool):
    """Pull an artifact from the Analysis Zoo."""
    try:
        client = ZooClient()

        # Get artifact info
        console.print(f"[cyan]Fetching artifact {artifact_id}...[/cyan]")
        artifact_info = client.get_artifact(artifact_id)

        # Determine target path
        if not target:
            target = ZOO_LOCAL_CACHE / artifact_id

        if target.exists() and not force:
            if not Confirm.ask(f"Directory {target} already exists. Overwrite?"):
                console.print("[yellow]Pull cancelled[/yellow]")
                return

        # Download
        console.print(f"[cyan]Downloading to {target}...[/cyan]")
        downloaded_path = client.download_artifact(artifact_id, target)

        console.print(
            f"[green]✓ Successfully pulled {
                artifact_info['name']} to {downloaded_path}[/green]")

    except Exception as e:
        console.print(f"[red]Pull failed: {e}[/red]")
        raise click.Abort()


@zoo.command()
@click.option('--query', help='Search query')
@click.option('--type', 'artifact_type', help='Filter by artifact type')
@click.option('--tags', help='Filter by comma-separated tags')
@click.option('--model', help='Filter by model name')
@click.option('--limit', default=20, help='Maximum results')
@click.option('--json', 'output_json', is_flag=True, help='Output as JSON')
def search(
    query: str,
    artifact_type: str,
    tags: str,
    model: str,
    limit: int,
    output_json: bool
):
    """Search for artifacts in the Analysis Zoo."""
    try:
        client = ZooClient()

        # Parse tags
        tag_list = [tag.strip() for tag in tags.split(",")] if tags else None

        # Search
        console.print("[cyan]Searching...[/cyan]")
        results = client.search_artifacts(
            query=query,
            artifact_type=artifact_type,
            tags=tag_list,
            model_name=model,
            limit=limit
        )

        if output_json:
            console.print(json.dumps(results, indent=2, default=str))
            return

        # Display results
        artifacts = results.get('artifacts', [])
        total = results.get('total_count', 0)

        if not artifacts:
            console.print("[yellow]No artifacts found[/yellow]")
            return

        console.print(f"\n[bold]Found {len(artifacts)} of {total} artifacts:[/bold]\n")

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("ID", style="cyan", width=8)
        table.add_column("Name", style="green")
        table.add_column("Type", style="blue")
        table.add_column("Version", style="yellow")
        table.add_column("Author", style="magenta")
        table.add_column("Stars", style="bright_yellow", justify="right")
        table.add_column("Description", style="white", max_width=40)

        for artifact in artifacts:
            table.add_row(
                artifact['uuid'][:8],
                artifact['name'],
                artifact['artifact_type'],
                artifact['version'],
                artifact['authors'][0]['name'] if artifact['authors'] else 'N/A',
                str(artifact.get('stars', 0)),
                artifact['description'][:60] + "..." if len(artifact['description']) > 60 else artifact['description']
            )

        console.print(table)

        if len(artifacts) < total:
            console.print(f"\n[dim]Showing {len(artifacts)} of {total} results. Use --limit to see more.[/dim]")

    except Exception as e:
        console.print(f"[red]Search failed: {e}[/red]")
        raise click.Abort()


@zoo.command()
def status():
    """Show Analysis Zoo status and configuration."""
    try:
        client = ZooClient()

        console.print("[bold]Analysis Zoo Status[/bold]\n")

        # API status
        try:
            response = client.session.get(f"{client.api_url}/health")
            if response.status_code == 200:
                console.print("[green]✓ API server is online[/green]")
            else:
                console.print(
                    f"[red]✗ API server returned {
                        response.status_code}[/red]")
        except Exception:
            console.print("[red]✗ Cannot connect to API server[/red]")

        # Authentication status
        if client.token:
            console.print("[green]✓ Authenticated[/green]")
        else:
            console.print(
                "[yellow]! Not authenticated (use 'neuronmap zoo login')[/yellow]")

        # Configuration
        console.print(f"\n[bold]Configuration:[/bold]")
        console.print(f"API URL: {client.api_url}")
        console.print(f"Cache directory: {ZOO_LOCAL_CACHE}")

        # Cache status
        if ZOO_LOCAL_CACHE.exists():
            cached_items = list(ZOO_LOCAL_CACHE.iterdir())
            console.print(f"Cached artifacts: {len(cached_items)}")
        else:
            console.print("Cached artifacts: 0")

    except Exception as e:
        console.print(f"[red]Status check failed: {e}[/red]")
        raise click.Abort()


# Register commands with main CLI
def register_zoo_commands(cli_group):
    """Register Zoo commands with the main CLI group."""
    cli_group.add_command(zoo)
