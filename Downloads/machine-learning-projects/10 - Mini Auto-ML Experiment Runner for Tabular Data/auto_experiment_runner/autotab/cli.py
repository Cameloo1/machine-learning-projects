import argparse
import logging
from rich.console import Console
from rich.table import Table
from rich.logging import RichHandler
from autotab.config import load_config
from autotab.data import get_dataset_metadata, load_dataset
from autotab.runner import run_and_save_experiment, build_leaderboard
from pydantic import ValidationError

def main():
    parser = argparse.ArgumentParser(description="Mini Auto-ML Experiment Runner")
    parser.add_argument("--config", required=True, help="Path to the configuration YAML file")
    args = parser.parse_args()

    console = Console()

    logging.basicConfig(
        level="INFO",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, show_time=False, rich_tracebacks=True)],
    )

    try:
        config = load_config(args.config)
        console.print(f"[green]Successfully loaded config![/green]")
        console.print(f"Problem Name: [bold]{config.task.problem_name}[/bold]")
        console.print(f"Task Type: [bold]{config.task.type}[/bold]")
        
        # Get dataset metadata for display
        dataset = load_dataset(config)
        metadata = get_dataset_metadata(dataset)
        console.print(f"\n[bold]Dataset Metadata:[/bold]")
        console.print(f"  Rows: {metadata.n_rows}")
        console.print(f"  Columns: {metadata.n_cols}")
        console.print(f"  Numeric features: {metadata.n_numeric}")
        console.print(f"  Categorical features: {metadata.n_categorical}")
        
        # Count enabled models
        enabled_models = [m for m in config.models if m.enabled]
        console.print(f"\n[bold]Experiment Configuration:[/bold]")
        console.print(f"  Models to train: {len(enabled_models)}")
        console.print(f"  Primary metric: {config.evaluation.primary_metric}")
        
        # Run the experiment and save artifacts
        console.print(f"\n[bold cyan]Running experiment...[/bold cyan]")
        results, output_dir = run_and_save_experiment(config)
        console.print(f"[green][OK] Experiment complete! Trained {len(results)} models.[/green]")
        console.print(f"[green][OK] All artifacts saved to: {output_dir.absolute()}[/green]")
        
        # Build leaderboard for display
        leaderboard = build_leaderboard(results, config.evaluation.primary_metric)
        
        # Display leaderboard as a rich table
        console.print(f"\n[bold]Leaderboard (sorted by {config.evaluation.primary_metric}):[/bold]")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Rank", style="dim", width=6)
        table.add_column("Model", style="cyan")
        table.add_column("Train Time (s)", justify="right")
        
        # Add metric columns
        if len(results) > 0:
            for metric_name in results[0].metrics.keys():
                table.add_column(metric_name, justify="right")
        
        # Add rows
        for idx, row in leaderboard.iterrows():
            rank = str(idx + 1)
            model_name = row['model']
            train_time = f"{row['train_time_sec']:.3f}"
            
            # Get metric values
            metric_values = []
            if len(results) > 0:
                for metric_name in results[0].metrics.keys():
                    value = row.get(metric_name)
                    if value is not None:
                        metric_values.append(f"{value:.4f}")
                    else:
                        metric_values.append("N/A")
            
            table.add_row(rank, model_name, train_time, *metric_values)
        
        console.print(table)
        
        # Display best model
        if len(leaderboard) > 0:
            best_model = leaderboard.iloc[0]['model']
            best_score = leaderboard.iloc[0][config.evaluation.primary_metric]
            console.print(f"\n[bold green]Best Model: {best_model}[/bold green]")
            console.print(f"   {config.evaluation.primary_metric}: {best_score:.4f}")

    except FileNotFoundError:
        console.print(f"[red]Error: Config file not found at {args.config}[/red]")
    except ValidationError as e:
        console.print(f"[red]Config Validation Error:[/red]")
        console.print(e)
    except Exception as e:
        console.print(f"[red]An unexpected error occurred:[/red] {e}")
        raise

if __name__ == "__main__":
    main()
