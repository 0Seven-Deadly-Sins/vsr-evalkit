import argparse
import json
import sys

from .config import Config
from .backends import preflight_backends
from .errors import EvaluationError
from .offline import catalog, check_artifacts
from .registry import METRICS
from .runner import build_jobs, environment, evaluate


def identifiers(value):
    result = [v.strip() for v in value.split(",") if v.strip()]
    if not result:
        raise argparse.ArgumentTypeError("Supply at least one ID")
    return list(dict.fromkeys(result))


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="vsr-eval",
        description="Evaluate pre-generated VSR outputs; never download weights",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "doctor"):
        sub = commands.add_parser(command)
        sub.add_argument("--config", default="eval.local.yaml")
        sub.add_argument("--model", type=identifiers, required=True)
        sub.add_argument("--dataset", type=identifiers, required=True)
        sub.add_argument(
            "--metrics", help="comma-separated IDs; defaults to config / auto"
        )
        sub.add_argument("--protocol", help="explicit protocol preset override")
        sub.add_argument("--device", help="cpu, cuda or cuda:N")
        if command == "run":
            sub.add_argument("--output", help="new directory; must not already exist")
    sub = commands.add_parser("list")
    sub.add_argument("--config", help="include registered models/datasets")
    sub = commands.add_parser("weights")
    sub.add_argument("action", choices=["info"])
    sub.add_argument("--metric", choices=list(METRICS))
    args = parser.parse_args(argv)
    try:
        if args.command == "list":
            output = {"metrics": {name: vars(spec) for name, spec in METRICS.items()}}
            if args.config:
                config = Config(args.config)
                output.update(
                    models=list(config.data["models"]),
                    datasets=list(config.data["datasets"]),
                )
            print(json.dumps(output, indent=2, ensure_ascii=False))
        elif args.command == "weights":
            data = catalog()
            if args.metric:
                data = {key: data[key] for key in METRICS[args.metric].artifacts}
            print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            config = Config(args.config)
            if args.command == "doctor":
                jobs = build_jobs(
                    config, args.model, args.dataset, args.metrics, args.protocol
                )
                names = {n for job in jobs for n in job["metrics"]}
                check_artifacts(names, config.artifact_paths())
                runtime = config.runtime()
                if args.device:
                    runtime["device"] = args.device
                preflight_backends(names, runtime)
                print(
                    json.dumps(
                        {
                            "environment": environment(),
                            "jobs": [
                                {
                                    "model": j["model"],
                                    "dataset": j["dataset"],
                                    "clips": len(j["clips"]),
                                    "metrics": j["metrics"],
                                    "protocol": j["protocol"]["id"],
                                }
                                for j in jobs
                            ],
                            "note": "Input/artifact checks only; no model was loaded and no numerical validation is implied.",
                        },
                        indent=2,
                    )
                )
            else:
                report, destination = evaluate(
                    config,
                    args.model,
                    args.dataset,
                    metrics=args.metrics,
                    protocol=args.protocol,
                    device=args.device,
                    output=args.output,
                    progress=lambda message: print(
                        message, file=sys.stderr, flush=True
                    ),
                )
                print(f"{report['status']}: {destination}")
                return 0 if report["status"] == "complete" else 1
    except (EvaluationError, ValueError, KeyError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
