#!/usr/bin/env python3
"""
Script to safely remove large factories from HuggingFace dataset cache.

This script:
1. Creates a backup manifest of what will be deleted
2. Identifies all blobs associated with the target factories
3. Removes factory directories and orphaned blobs
4. Reports disk space freed

Usage:
    python scripts/remove_factories.py --dry-run  # Preview what will be deleted
    python scripts/remove_factories.py            # Actually delete
"""

import argparse
import os
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Set, Tuple


# Configuration
CACHE_BASE = Path("/scratch-shared/scur1900/huggingface_cache")
DATASET_PATH = CACHE_BASE / "hub/datasets--builddotai--Egocentric-10K"
SNAPSHOT_PATH = DATASET_PATH / "snapshots/d5c3224f6ed4bcb7dfc93f6fbdcd8890e6d67845"
BLOBS_PATH = DATASET_PATH / "blobs"

# Factories to remove (7 largest, ~4.2TB)
FACTORIES_TO_REMOVE = [
    "factory_014",
    "factory_046",
    "factory_047",
    "factory_013",
    "factory_027",
    "factory_026",
    "factory_029",
]


def get_blob_from_symlink(symlink_path: Path) -> str:
    """Extract blob hash from symlink target."""
    if symlink_path.is_symlink():
        target = os.readlink(symlink_path)
        blob_name = Path(target).name
        return blob_name
    return None


def collect_factory_blobs(factory_path: Path) -> Set[str]:
    """Collect all blob hashes referenced by a factory."""
    blobs = set()

    # Find all symlinks in the factory directory
    for root, dirs, files in os.walk(factory_path):
        for file in files:
            file_path = Path(root) / file
            if file_path.is_symlink():
                blob = get_blob_from_symlink(file_path)
                if blob:
                    blobs.add(blob)

    return blobs


def get_all_remaining_blobs(exclude_factories: List[str]) -> Set[str]:
    """Get all blobs referenced by factories NOT being removed."""
    remaining_blobs = set()

    for factory_dir in SNAPSHOT_PATH.iterdir():
        if factory_dir.is_dir() and factory_dir.name.startswith("factory_"):
            if factory_dir.name not in exclude_factories:
                factory_blobs = collect_factory_blobs(factory_dir)
                remaining_blobs.update(factory_blobs)
                print(f"  Keeping {factory_dir.name}: {len(factory_blobs)} blobs")

    return remaining_blobs


def get_blob_size(blob_hash: str) -> int:
    """Get size of a blob file in bytes."""
    blob_path = BLOBS_PATH / blob_hash
    if blob_path.exists():
        return blob_path.stat().st_size
    return 0


def create_backup_manifest(
    factories: List[str],
    factory_blobs: dict,
    orphaned_blobs: Set[str],
    output_dir: Path
):
    """Create a JSON manifest of what will be deleted."""
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "factories_to_remove": factories,
        "factory_blob_mapping": {
            factory: list(blobs) for factory, blobs in factory_blobs.items()
        },
        "orphaned_blobs": list(orphaned_blobs),
        "total_blobs_to_remove": len(orphaned_blobs),
        "paths": {
            "snapshot_path": str(SNAPSHOT_PATH),
            "blobs_path": str(BLOBS_PATH),
        }
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / f"deletion_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"\nBackup manifest saved to: {manifest_path}")
    return manifest_path


def format_size(bytes_size: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"


def main():
    parser = argparse.ArgumentParser(description="Remove large factories from HF dataset cache")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview what will be deleted without actually deleting"
    )
    parser.add_argument(
        "--backup-dir",
        type=str,
        default="/gpfs/home4/scur1900/cosmos-finetune/backups",
        help="Directory to save deletion manifest"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Factory Removal Script")
    print("=" * 80)
    print(f"Mode: {'DRY RUN (no files will be deleted)' if args.dry_run else 'DELETION MODE'}")
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Snapshot path: {SNAPSHOT_PATH}")
    print(f"Blobs path: {BLOBS_PATH}")
    print()

    # Validate paths
    if not SNAPSHOT_PATH.exists():
        print(f"ERROR: Snapshot path does not exist: {SNAPSHOT_PATH}")
        return 1

    if not BLOBS_PATH.exists():
        print(f"ERROR: Blobs path does not exist: {BLOBS_PATH}")
        return 1

    # Step 1: Collect blobs from factories to be removed
    print("Step 1: Collecting blobs from factories to be removed...")
    factory_blobs = {}
    total_factory_blobs = set()

    for factory_name in FACTORIES_TO_REMOVE:
        factory_path = SNAPSHOT_PATH / factory_name
        if not factory_path.exists():
            print(f"  WARNING: {factory_name} does not exist, skipping")
            continue

        blobs = collect_factory_blobs(factory_path)
        factory_blobs[factory_name] = blobs
        total_factory_blobs.update(blobs)
        print(f"  {factory_name}: {len(blobs)} blobs")

    print(f"\nTotal unique blobs in factories to remove: {len(total_factory_blobs)}")

    # Step 2: Collect blobs from remaining factories
    print("\nStep 2: Collecting blobs from remaining factories...")
    remaining_blobs = get_all_remaining_blobs(FACTORIES_TO_REMOVE)
    print(f"\nTotal unique blobs in remaining factories: {len(remaining_blobs)}")

    # Step 3: Identify orphaned blobs (safe to delete)
    print("\nStep 3: Identifying orphaned blobs...")
    orphaned_blobs = total_factory_blobs - remaining_blobs
    shared_blobs = total_factory_blobs & remaining_blobs

    print(f"  Orphaned blobs (will be deleted): {len(orphaned_blobs)}")
    print(f"  Shared blobs (will be kept): {len(shared_blobs)}")

    # Step 4: Calculate disk space to be freed
    print("\nStep 4: Calculating disk space to be freed...")
    total_size = 0
    blob_count = 0

    for blob_hash in orphaned_blobs:
        size = get_blob_size(blob_hash)
        total_size += size
        if size > 0:
            blob_count += 1

    print(f"  Total disk space to free: {format_size(total_size)}")
    print(f"  Number of blob files to delete: {blob_count}")

    # Step 5: Create backup manifest
    print("\nStep 5: Creating backup manifest...")
    manifest_path = create_backup_manifest(
        FACTORIES_TO_REMOVE,
        factory_blobs,
        orphaned_blobs,
        Path(args.backup_dir)
    )

    # Step 6: Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Factories to remove: {len(FACTORIES_TO_REMOVE)}")
    for factory in FACTORIES_TO_REMOVE:
        if factory in factory_blobs:
            print(f"  - {factory} ({len(factory_blobs[factory])} blobs)")
    print(f"\nOrphaned blobs to delete: {len(orphaned_blobs)}")
    print(f"Shared blobs to keep: {len(shared_blobs)}")
    print(f"Disk space to free: {format_size(total_size)}")
    print(f"\nBackup manifest: {manifest_path}")
    print("=" * 80)

    # Step 7: Perform deletion
    if args.dry_run:
        print("\nDRY RUN MODE - No files were deleted")
        print("Run without --dry-run to actually delete the files")
        return 0

    print("\nStep 6: Performing deletion...")
    response = input(f"\nAre you sure you want to delete {len(FACTORIES_TO_REMOVE)} factories "
                     f"and {len(orphaned_blobs)} blobs (~{format_size(total_size)})? [yes/NO]: ")

    if response.lower() != 'yes':
        print("Deletion cancelled")
        return 0

    print("\nDeleting factories...")
    deleted_factories = 0
    for factory_name in FACTORIES_TO_REMOVE:
        factory_path = SNAPSHOT_PATH / factory_name
        if factory_path.exists():
            try:
                subprocess.run(["rm", "-rf", str(factory_path)], check=True)
                print(f"  ✓ Deleted {factory_name}")
                deleted_factories += 1
            except subprocess.CalledProcessError as e:
                print(f"  ✗ Failed to delete {factory_name}: {e}")

    print(f"\nDeleting {len(orphaned_blobs)} orphaned blobs...")
    deleted_blobs = 0
    failed_blobs = 0

    for i, blob_hash in enumerate(orphaned_blobs, 1):
        blob_path = BLOBS_PATH / blob_hash
        if blob_path.exists():
            try:
                blob_path.unlink()
                deleted_blobs += 1
                if i % 100 == 0:
                    print(f"  Progress: {i}/{len(orphaned_blobs)} blobs deleted")
            except OSError as e:
                print(f"  ✗ Failed to delete blob {blob_hash}: {e}")
                failed_blobs += 1

    print("\n" + "=" * 80)
    print("DELETION COMPLETE")
    print("=" * 80)
    print(f"Factories deleted: {deleted_factories}/{len(FACTORIES_TO_REMOVE)}")
    print(f"Blobs deleted: {deleted_blobs}/{len(orphaned_blobs)}")
    if failed_blobs > 0:
        print(f"Failed deletions: {failed_blobs}")
    print(f"Disk space freed: {format_size(total_size)}")
    print(f"\nManifest saved to: {manifest_path}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    exit(main())
