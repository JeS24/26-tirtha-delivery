#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import fire


DEFAULTS = {
    "ratio": "0.5",
    "k": "32",
    "merge_cap": "0.5",
    "opacity_threshold": "0.1",
    "lam_geo": "1.0",
    "lam_sh": "1.0",
    "gpu": "0",
    "voxel_size": "0.0125",
    "collision": "faces",
    "lod_levels": 4,
    "nanogs_python": "python",
}


class SplatPipeline:
    def __init__(self):
        self.logger = logging.getLogger("splat_pipeline")
        self.logger.setLevel(logging.DEBUG)

        self.input_path: Path | None = None
        self.scene: str | None = None
        self.outdir: Path | None = None
        self.scene_dir: Path | None = None
        self.nanogs_dir: Path | None = None
        self.current_input: Path | None = None
        self.results_csv: Path | None = None
        self.dirs: dict[str, Path] = {}

        self.ratio = DEFAULTS["ratio"]
        self.k = DEFAULTS["k"]
        self.merge_cap = DEFAULTS["merge_cap"]
        self.opacity_threshold = DEFAULTS["opacity_threshold"]
        self.lam_geo = DEFAULTS["lam_geo"]
        self.lam_sh = DEFAULTS["lam_sh"]
        self.gpu = DEFAULTS["gpu"]
        self.voxel_size = DEFAULTS["voxel_size"]
        self.collision = DEFAULTS["collision"]
        self.lod_levels = DEFAULTS["lod_levels"]
        self.nanogs_python = DEFAULTS["nanogs_python"]

        self.resume = False
        self.dry_run = False

    # -------------------------
    # Setup
    # -------------------------

    def configure(
        self,
        input_ply: str,
        scene: str,
        outdir: str = "outputs",
        nanogs_dir: str | None = None,
        ratio: str = DEFAULTS["ratio"],
        k: str = DEFAULTS["k"],
        gpu: str = DEFAULTS["gpu"],
        voxel_size: str = DEFAULTS["voxel_size"],
        collision: str = DEFAULTS["collision"],
        lod_levels: int = DEFAULTS["lod_levels"],
        nanogs_python: str = DEFAULTS["nanogs_python"],
        resume: bool = False,
        dry_run: bool = False,
        verbose: bool = False,
    ) -> None:
        if collision not in ["faces", "smooth"]:
            raise SystemExit("ERROR: collision must be either 'faces' or 'smooth'")

        if nanogs_dir is None:
            nanogs_dir = os.environ.get("NANOGS_DIR")

        if not nanogs_dir:
            raise SystemExit("ERROR: nanogs_dir was not provided and NANOGS_DIR is not set.")

        self.input_path = Path(input_ply).expanduser().resolve()
        self.scene = scene
        self.outdir = Path(outdir).expanduser().resolve()
        self.scene_dir = self.outdir / scene
        self.nanogs_dir = Path(nanogs_dir).expanduser().resolve()
        self.current_input = self.input_path
        self.results_csv = self.scene_dir / "results.csv"

        self.ratio = ratio
        self.k = k
        self.gpu = gpu
        self.voxel_size = voxel_size
        self.collision = collision
        self.lod_levels = int(lod_levels)
        self.nanogs_python = nanogs_python

        self.resume = resume
        self.dry_run = dry_run

        self.validate_paths()
        self.dirs = self.make_dirs()
        self.setup_logging(verbose)

        self.logger.info("Final selected parameters:")
        self.logger.info("NanoGS: r=%s, k=%s", self.ratio, self.k)
        self.logger.info("NanoGS Python: %s", self.nanogs_python)
        self.logger.info("splat-transform GPU: %s", self.gpu)
        self.logger.info("voxel size: %s", self.voxel_size)
        self.logger.info("collision mesh: %s", self.collision)
        self.logger.info("LoD levels: %s", self.lod_levels)

    def validate_paths(self) -> None:
        assert self.input_path is not None
        assert self.nanogs_dir is not None

        self.check_command_exists("splat-transform")
        self.check_command_exists(self.nanogs_python)

        if not self.input_path.exists():
            raise SystemExit(f"ERROR: Input file not found: {self.input_path}")

        if not self.nanogs_dir.exists():
            raise SystemExit(f"ERROR: NanoGS directory not found: {self.nanogs_dir}")

    def setup_logging(self, verbose: bool = False) -> None:
        assert self.scene_dir is not None

        self.logger.handlers.clear()

        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.DEBUG if verbose else logging.INFO)
        console.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
        )
        self.logger.addHandler(console)

        log_dir = self.scene_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_dir / "pipeline.log")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        self.logger.addHandler(file_handler)

    def make_dirs(self) -> dict[str, Path]:
        assert self.scene_dir is not None

        dirs = {
            "nanogs": self.scene_dir / "01_nanogs",
            "sog": self.scene_dir / "02_sog",
            "filters": self.scene_dir / "03_filters",
            "lod": self.scene_dir / "04_lod",
            "voxel": self.scene_dir / "05_voxel",
            "collision": self.scene_dir / "06_collision",
            "logs": self.scene_dir / "logs",
        }

        for folder in dirs.values():
            folder.mkdir(parents=True, exist_ok=True)

        return dirs

    # -------------------------
    # Utilities
    # -------------------------

    def check_command_exists(self, command: str) -> None:
        if shutil.which(command) is None:
            raise SystemExit(f"ERROR: Required command not found: {command}")

    def format_size(self, size_bytes: int) -> str:
        if size_bytes < 1024:
            return f"{size_bytes} B"
        if size_bytes < 1024**2:
            return f"{size_bytes / 1024:.2f} KB"
        if size_bytes < 1024**3:
            return f"{size_bytes / (1024**2):.2f} MB"
        return f"{size_bytes / (1024**3):.2f} GB"

    def format_seconds(self, seconds: float) -> str:
        if seconds <= 0:
            return "-"
        if seconds < 60:
            return f"{seconds:.2f}s"
        return f"{int(seconds // 60)}m {seconds % 60:.2f}s"

    def file_size_mb(self, path: Path) -> float:
        return path.stat().st_size / (1024 * 1024) if path.exists() else 0.0

    def folder_size(self, path: Path) -> int:
        if not path.exists():
            return 0
        return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())

    def ply_vertex_count(self, path: Path) -> str:
        if not path.exists() or path.suffix.lower() != ".ply":
            return ""

        try:
            with open(path, "rb") as f:
                for _ in range(150):
                    line = f.readline().decode("utf-8", errors="ignore").strip()
                    if line.startswith("element vertex"):
                        return line.split()[-1]
                    if line == "end_header":
                        break
        except Exception:
            return ""

        return ""

    def command_version(self, command: str) -> str:
        for flag in ["--version", "-v"]:
            try:
                result = subprocess.run(
                    [command, flag],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=10,
                )
                if result.stdout.strip():
                    return result.stdout.strip().splitlines()[0]
            except Exception:
                pass
        return "unknown"

    def require_output(self, path: Path) -> None:
        if not path.exists():
            raise SystemExit(f"ERROR: Expected output was not created: {path}")
        if path.stat().st_size == 0:
            raise SystemExit(f"ERROR: Output file is empty: {path}")

    # -------------------------
    # Command execution
    # -------------------------

    def append_result(
        self,
        stage: str,
        command: list[str],
        input_path: Path,
        output_path: Path,
        elapsed: float,
        status: str,
        return_code: int | str,
    ) -> None:
        assert self.results_csv is not None

        file_exists = self.results_csv.exists()

        with open(self.results_csv, "a", newline="") as f:
            writer = csv.writer(f)

            if not file_exists:
                writer.writerow(
                    [
                        "timestamp",
                        "stage",
                        "status",
                        "return_code",
                        "elapsed_seconds",
                        "input_file",
                        "input_size_mb",
                        "input_vertices",
                        "output_file",
                        "output_size_mb",
                        "output_vertices",
                        "command",
                    ]
                )

            writer.writerow(
                [
                    datetime.now().isoformat(timespec="seconds"),
                    stage,
                    status,
                    return_code,
                    f"{elapsed:.2f}",
                    str(input_path),
                    f"{self.file_size_mb(input_path):.2f}",
                    self.ply_vertex_count(input_path),
                    str(output_path),
                    f"{self.file_size_mb(output_path):.2f}",
                    self.ply_vertex_count(output_path),
                    " ".join(str(x) for x in command),
                ]
            )

    def run_cmd(
        self,
        stage: str,
        cmd: list[str],
        input_path: Path,
        output_path: Path,
        log_file: Path,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        assert self.results_csv is not None

        printable = " ".join(str(x) for x in cmd)

        if self.resume and output_path.exists() and output_path.stat().st_size > 1024:
            self.logger.info("[SKIP] %s already exists: %s", stage, output_path)
            self.append_result(stage, cmd, input_path, output_path, 0.0, "skipped_resume", "SKIP")
            return

        if not self.resume and output_path.exists():
            self.logger.info("Removing old output before rerun: %s", output_path)
            output_path.unlink()

        self.logger.info("=" * 70)
        self.logger.info("[%s] Running", stage)
        self.logger.info("%s", printable)
        self.logger.info("=" * 70)

        if self.dry_run:
            self.append_result(stage, cmd, input_path, output_path, 0.0, "dry_run", "DRY_RUN")
            self.logger.info("[dry-run] Command not executed.")
            return

        start = time.time()

        with open(log_file, "w") as log:
            log.write(f"Stage: {stage}\n")
            log.write(f"Started: {datetime.now().isoformat(timespec='seconds')}\n")
            log.write(f"Working directory: {cwd if cwd else Path.cwd()}\n")
            log.write(f"Command: {printable}\n\n")
            log.flush()

            process = subprocess.run(
                cmd,
                cwd=cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )

        elapsed = time.time() - start
        status = "success" if process.returncode == 0 else "failed"

        with open(log_file, "a") as log:
            log.write(f"\nFinished: {datetime.now().isoformat(timespec='seconds')}\n")
            log.write(f"Elapsed seconds: {elapsed:.2f}\n")
            log.write(f"Return code: {process.returncode}\n")

        self.append_result(stage, cmd, input_path, output_path, elapsed, status, process.returncode)

        if process.returncode != 0:
            self.logger.error("[%s] FAILED. Check log: %s", stage, log_file)
            raise SystemExit(1)

        self.require_output(output_path)

        self.logger.info(
            "[%s] Done in %.2fs | Size: %s",
            stage,
            elapsed,
            self.format_size(output_path.stat().st_size),
        )

    # -------------------------
    # Pipeline stages
    # -------------------------

    def run_nanogs(self) -> None:
        assert self.input_path is not None
        assert self.current_input is not None
        assert self.nanogs_dir is not None
        assert self.scene is not None

        output = self.dirs["nanogs"] / f"{self.scene}_nanogs_r{self.ratio.replace('.', '')}_k{self.k}.ply"

        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(filter(None, ["src", env.get("PYTHONPATH")]))

        cmd = [
            self.nanogs_python,
            "-m",
            "nanogs.simplification",
            "--ply",
            str(self.input_path),
            "-o",
            str(output),
            "-r",
            str(self.ratio),
            "--k",
            str(self.k),
            "--merge_cap",
            self.merge_cap,
            "--opacity_threshold",
            self.opacity_threshold,
            "--lam_geo",
            self.lam_geo,
            "--lam_sh",
            self.lam_sh,
        ]

        self.run_cmd(
            stage="01_nanogs",
            cmd=cmd,
            input_path=self.input_path,
            output_path=output,
            log_file=self.dirs["logs"] / "01_nanogs.log",
            cwd=self.nanogs_dir,
            env=env,
        )

        self.current_input = output

    def run_raw_sog(self) -> None:
        self.run_splat_stage("02_raw_sog", self.dirs["sog"] / f"{self.scene}_raw.sog")

    def run_floater_sog(self) -> None:
        self.run_splat_stage(
            "03_floater_sog",
            self.dirs["filters"] / f"{self.scene}_floater_filtered.sog",
            ["-G"],
        )

    def run_cluster_sog(self) -> None:
        self.run_splat_stage(
            "04_cluster_sog",
            self.dirs["filters"] / f"{self.scene}_cluster_filtered.sog",
            ["-D"],
        )

    def run_combined_sog(self) -> None:
        self.run_splat_stage(
            "05_combined_filter_sog",
            self.dirs["filters"] / f"{self.scene}_combined_filtered.sog",
            ["-G", "-D"],
        )

    def run_lods(self) -> None:
        for level in range(self.lod_levels):
            self.run_splat_stage(
                f"06_lod{level}",
                self.dirs["lod"] / f"{self.scene}_lod{level}.ply",
                ["-l", str(level)],
            )

    def run_voxel(self) -> None:
        voxel_name = self.voxel_size.replace(".", "")
        self.run_splat_stage(
            "07_voxel",
            self.dirs["voxel"] / f"{self.scene}_voxel_{voxel_name}.voxel.json",
            ["--voxel-params", self.voxel_size],
        )

    def run_collision_mesh(self) -> None:
        self.run_splat_stage(
            "08_collision_mesh",
            self.dirs["collision"] / f"{self.scene}_{self.collision}.collision.glb",
            ["-K", self.collision],
        )

    def run_splat_stage(
        self,
        stage: str,
        output_path: Path,
        extra_flags: list[str] | None = None,
    ) -> None:
        assert self.current_input is not None

        cmd = [
            "splat-transform",
            "-w",
            "-g",
            self.gpu,
            str(self.current_input),
            *(extra_flags or []),
            str(output_path),
        ]

        self.run_cmd(
            stage=stage,
            cmd=cmd,
            input_path=self.current_input,
            output_path=output_path,
            log_file=self.dirs["logs"] / f"{stage}.log",
        )

    # -------------------------
    # Reports
    # -------------------------

    def read_results_csv(self) -> list[dict[str, str]]:
        assert self.results_csv is not None

        if not self.results_csv.exists():
            return []

        with open(self.results_csv, newline="") as f:
            return list(csv.DictReader(f))

    def latest_rows_by_stage(self) -> dict[str, dict[str, str]]:
        latest = {}
        for row in self.read_results_csv():
            latest[row["stage"]] = row
        return latest

    def write_comparison_report(self) -> None:
        assert self.scene_dir is not None
        assert self.input_path is not None
        assert self.scene is not None

        latest = self.latest_rows_by_stage()
        report_path = self.scene_dir / "comparison_report.md"

        original_size = self.file_size_mb(self.input_path)
        original_vertices = self.ply_vertex_count(self.input_path)

        stages_order = [
            "01_nanogs",
            "02_raw_sog",
            "03_floater_sog",
            "04_cluster_sog",
            "05_combined_filter_sog",
            "06_lod0",
            "06_lod1",
            "06_lod2",
            "06_lod3",
            "07_voxel",
            "08_collision_mesh",
        ]

        stage_names = {
            "01_nanogs": "NanoGS simplified PLY",
            "02_raw_sog": "Raw SOG",
            "03_floater_sog": "Floater-filtered SOG",
            "04_cluster_sog": "Cluster-filtered SOG",
            "05_combined_filter_sog": "Combined-filtered SOG",
            "06_lod0": "LoD0 PLY",
            "06_lod1": "LoD1 PLY",
            "06_lod2": "LoD2 PLY",
            "06_lod3": "LoD3 PLY",
            "07_voxel": "Voxel JSON",
            "08_collision_mesh": "Collision GLB",
        }

        with open(report_path, "w") as f:
            f.write(f"# Comparison Report: {self.scene}\n\n")
            f.write(f"Generated: {datetime.now().isoformat(timespec='seconds')}\n\n")

            f.write("## Selected Parameters\n\n")
            f.write(f"- NanoGS ratio: `{self.ratio}`\n")
            f.write(f"- NanoGS k: `{self.k}`\n")
            f.write(f"- Voxel size: `{self.voxel_size}`\n")
            f.write(f"- Collision mesh: `{self.collision}`\n")
            f.write(f"- LoD levels: `{self.lod_levels}`\n\n")

            f.write("## Size, Vertex Count, and Runtime Comparison\n\n")
            f.write("| Stage | Status | Size | Vertices | Runtime | Compression vs Original |\n")
            f.write("|---|---:|---:|---:|---:|---:|\n")
            f.write(
                f"| Original PLY | input | {original_size:.2f} MB | "
                f"{original_vertices or '-'} | - | 1.00x |\n"
            )

            for stage in stages_order:
                row = latest.get(stage)
                if not row:
                    continue

                output_size = float(row.get("output_size_mb") or 0)
                vertices = row.get("output_vertices") or "-"
                elapsed = float(row.get("elapsed_seconds") or 0)
                status = row.get("status", "-")
                compression = f"{original_size / output_size:.2f}x" if output_size > 0 else "-"

                f.write(
                    f"| {stage_names.get(stage, stage)} | {status} | "
                    f"{output_size:.2f} MB | {vertices} | "
                    f"{self.format_seconds(elapsed)} | {compression} |\n"
                )

            f.write("\n## Notes\n\n")
            f.write("- Compression = original input PLY size / stage output size.\n")
            f.write("- Vertex count is available only for PLY outputs.\n")
            f.write("- Resume-skipped stages show runtime as `-`.\n")

        self.logger.info("Comparison report written to: %s", report_path)

    def write_metadata(self) -> None:
        assert self.scene_dir is not None
        assert self.input_path is not None
        assert self.scene is not None

        metadata = {
            "scene": self.scene,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "input": str(self.input_path),
            "nanogs_ratio": self.ratio,
            "nanogs_k": self.k,
            "merge_cap": self.merge_cap,
            "opacity_threshold": self.opacity_threshold,
            "lam_geo": self.lam_geo,
            "lam_sh": self.lam_sh,
            "gpu": self.gpu,
            "voxel_size": self.voxel_size,
            "collision": self.collision,
            "lod_levels": self.lod_levels,
            "cli_python_version": sys.version.split()[0],
            "nanogs_python": self.nanogs_python,
            "splat_transform_version": self.command_version("splat-transform"),
        }

        path = self.scene_dir / "metadata.json"
        with open(path, "w") as f:
            json.dump(metadata, f, indent=4)

        self.logger.info("Metadata written to: %s", path)

    def write_summary(self) -> None:
        assert self.scene_dir is not None
        assert self.input_path is not None
        assert self.current_input is not None
        assert self.nanogs_dir is not None
        assert self.scene is not None

        path = self.scene_dir / "pipeline_summary.txt"

        with open(path, "w") as f:
            f.write("Final NanoGS + splat-transform Pipeline Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Scene: {self.scene}\n")
            f.write(f"Original input: {self.input_path}\n")
            f.write(f"Final processing input: {self.current_input}\n")
            f.write(f"Output folder: {self.scene_dir}\n")
            f.write(f"NanoGS directory: {self.nanogs_dir}\n")
            f.write(f"NanoGS Python: {self.nanogs_python}\n\n")

            f.write("Software versions:\n")
            f.write(f"CLI Python: {sys.version.split()[0]}\n")
            f.write(f"splat-transform: {self.command_version('splat-transform')}\n\n")

            f.write("Selected parameters:\n")
            f.write(f"NanoGS ratio: {self.ratio}\n")
            f.write(f"NanoGS k: {self.k}\n")
            f.write(f"merge_cap: {self.merge_cap}\n")
            f.write(f"opacity_threshold: {self.opacity_threshold}\n")
            f.write(f"lam_geo: {self.lam_geo}\n")
            f.write(f"lam_sh: {self.lam_sh}\n")
            f.write(f"splat-transform GPU: {self.gpu}\n")
            f.write(f"voxel size: {self.voxel_size}\n")
            f.write(f"collision mesh: {self.collision}\n")
            f.write(f"LoD levels: {self.lod_levels}\n\n")

            f.write("Generated files:\n")
            f.write(f"- results.csv: {self.scene_dir / 'results.csv'}\n")
            f.write(f"- comparison_report.md: {self.scene_dir / 'comparison_report.md'}\n")
            f.write(f"- metadata.json: {self.scene_dir / 'metadata.json'}\n")
            f.write(f"- pipeline_diagram.txt: {self.scene_dir / 'pipeline_diagram.txt'}\n")
            f.write(f"- pipeline log: {self.scene_dir / 'logs' / 'pipeline.log'}\n")

        self.logger.info("Summary written to: %s", path)

    def write_pipeline_diagram(self) -> None:
        assert self.scene_dir is not None

        diagram = """
Pipeline Diagram
================

Original PLY
     |
     v
NanoGS Simplification
     |
     v
Simplified PLY
     |
     +----------------------+
     |                      |
     v                      v
Raw SOG              Filtered SOGs
                           |
                           v
                    LoD Generation
                           |
             +-------------+-------------+
             |                           |
             v                           v
        Voxel JSON              Collision GLB
"""

        path = self.scene_dir / "pipeline_diagram.txt"
        with open(path, "w") as f:
            f.write(diagram)

        self.logger.info("Pipeline diagram written to: %s", path)

    def print_disk_usage(self) -> None:
        assert self.scene_dir is not None

        print("\n" + "=" * 60)
        print("DISK USAGE REPORT")
        print("=" * 60)

        total = 0

        for folder in [
            "01_nanogs",
            "02_sog",
            "03_filters",
            "04_lod",
            "05_voxel",
            "06_collision",
            "logs",
        ]:
            path = self.scene_dir / folder
            size = self.folder_size(path)
            total += size
            print(f"{folder:<15} {self.format_size(size)}")

        print("-" * 60)
        print(f"{'Total':<15} {self.format_size(total)}")
        print("=" * 60)

    def write_all_reports(self) -> None:
        self.write_summary()
        self.write_metadata()
        self.write_pipeline_diagram()
        self.write_comparison_report()
        self.print_disk_usage()

    # -------------------------
    # Fire CLI
    # -------------------------

    def run(
        self,
        input_ply: str,
        scene: str,
        outdir: str = "outputs",
        nanogs_dir: str | None = None,
        ratio: str = DEFAULTS["ratio"],
        k: str = DEFAULTS["k"],
        gpu: str = DEFAULTS["gpu"],
        voxel_size: str = DEFAULTS["voxel_size"],
        collision: str = DEFAULTS["collision"],
        lod_levels: int = DEFAULTS["lod_levels"],
        nanogs_python: str = DEFAULTS["nanogs_python"],
        nanogs: bool = False,
        raw_sog: bool = False,
        floater_sog: bool = False,
        cluster_sog: bool = False,
        combined_sog: bool = False,
        lod: bool = False,
        voxel: bool = False,
        collision_mesh: bool = False,
        all: bool = False,
        resume: bool = False,
        dry_run: bool = False,
        verbose: bool = False,
    ) -> None:
        self.configure(
            input_ply=input_ply,
            scene=scene,
            outdir=outdir,
            nanogs_dir=nanogs_dir,
            ratio=ratio,
            k=k,
            gpu=gpu,
            voxel_size=voxel_size,
            collision=collision,
            lod_levels=lod_levels,
            nanogs_python=nanogs_python,
            resume=resume,
            dry_run=dry_run,
            verbose=verbose,
        )

        try:
            if nanogs or all:
                self.run_nanogs()

            if raw_sog or all:
                self.run_raw_sog()

            if floater_sog or all:
                self.run_floater_sog()

            if cluster_sog or all:
                self.run_cluster_sog()

            if combined_sog or all:
                self.run_combined_sog()

            if lod or all:
                self.run_lods()

            if voxel or all:
                self.run_voxel()

            if collision_mesh or all:
                self.run_collision_mesh()

        finally:
            self.write_all_reports()

        self.logger.info("Pipeline completed.")
        self.logger.info("Scene output folder: %s", self.scene_dir)
        self.logger.info("CSV report: %s", self.results_csv)
        self.logger.info("Comparison report: %s", self.scene_dir / "comparison_report.md")


if __name__ == "__main__":
    fire.Fire(SplatPipeline)