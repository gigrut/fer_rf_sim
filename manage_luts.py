#!/usr/bin/env python3
"""
LUT Management Utility for FER Simulation

This script helps manage the expensive-to-compute LUT files by providing
utilities to list, clean up, and check compatibility of existing LUTs.
"""

import argparse
import os
from pathlib import Path
import h5py
import numpy as np

def list_luts(fer_output_dir='fer_output', show_details=True):
    """List all LUT files with their parameters and sizes."""
    lut_dir = Path(fer_output_dir)
    if not lut_dir.exists():
        print("No fer_output directory found.")
        return
    
    lut_files = list(lut_dir.glob("transmission_lut_*.h5"))
    if not lut_files:
        print("No LUT files found.")
        return
    
    total_size = 0
    print(f"Found {len(lut_files)} LUT file(s):")
    print("-" * 80)
    
    for lut_file in sorted(lut_files):
        try:
            with h5py.File(lut_file, 'r') as f:
                phi_tip = f.attrs.get('phi_tip', 'N/A')
                phi_samp = f.attrs.get('phi_samp', 'N/A')
                n_E = len(f['E']) if 'E' in f else 'N/A'
                n_V = len(f['V']) if 'V' in f else 'N/A'
                n_Z = len(f['z']) if 'z' in f else 'N/A'
                file_size = lut_file.stat().st_size / (1024*1024)  # MB
                total_size += file_size
                
                print(f"  {lut_file.name}")
                print(f"    Size: {file_size:.1f} MB")
                if show_details:
                    print(f"    Parameters: phi_tip={phi_tip}, phi_samp={phi_samp}")
                    print(f"    Grid: n_E={n_E}, n_V={n_V}, n_Z={n_Z}")
                print()
        except Exception as e:
            print(f"  {lut_file.name} (error reading: {e})")
            print()
    
    print(f"Total LUT storage: {total_size:.1f} MB")

def cleanup_old_luts(fer_output_dir='fer_output', keep_recent=3):
    """Keep only the most recent LUT files, delete older ones."""
    lut_dir = Path(fer_output_dir)
    if not lut_dir.exists():
        print("No fer_output directory found.")
        return
    
    lut_files = list(lut_dir.glob("transmission_lut_*.h5"))
    if len(lut_files) <= keep_recent:
        print(f"Only {len(lut_files)} LUT files found, keeping all.")
        return
    
    # Sort by modification time (newest first)
    lut_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
    
    print(f"Keeping {keep_recent} most recent LUT files:")
    for i, lut_file in enumerate(lut_files[:keep_recent]):
        size_mb = lut_file.stat().st_size / (1024*1024)
        print(f"  {lut_file.name} ({size_mb:.1f} MB)")
    
    files_to_delete = lut_files[keep_recent:]
    if files_to_delete:
        print(f"\nDeleting {len(files_to_delete)} older LUT files:")
        total_deleted_size = 0
        for lut_file in files_to_delete:
            size_mb = lut_file.stat().st_size / (1024*1024)
            total_deleted_size += size_mb
            print(f"  {lut_file.name} ({size_mb:.1f} MB)")
            lut_file.unlink()
        
        print(f"\nFreed {total_deleted_size:.1f} MB of disk space.")

def check_compatibility(phi_tip, phi_samp, n_E, n_V, n_Z, fer_output_dir='fer_output'):
    """Check if a compatible LUT exists for the given parameters."""
    lut_dir = Path(fer_output_dir)
    if not lut_dir.exists():
        print("No fer_output directory found.")
        return False
    
    lut_files = list(lut_dir.glob("transmission_lut_*.h5"))
    for lut_file in lut_files:
        try:
            with h5py.File(lut_file, 'r') as f:
                if (abs(f.attrs.get('phi_tip', 0) - phi_tip) < 1e-8 and
                    abs(f.attrs.get('phi_samp', 0) - phi_samp) < 1e-8 and
                    len(f['E']) == n_E and
                    len(f['V']) == n_V and
                    len(f['z']) == n_Z):
                    print(f"✓ Compatible LUT found: {lut_file.name}")
                    return True
        except Exception:
            continue
    
    print("✗ No compatible LUT found.")
    return False

def main():
    parser = argparse.ArgumentParser(description="Manage FER simulation LUT files")
    parser.add_argument('--fer-output', default='fer_output', help='FER output directory')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all LUT files')
    list_parser.add_argument('--brief', action='store_true', help='Show only file names and sizes')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Remove old LUT files')
    cleanup_parser.add_argument('--keep', type=int, default=3, help='Number of recent files to keep')
    
    # Check command
    check_parser = subparsers.add_parser('check', help='Check LUT compatibility')
    check_parser.add_argument('--phi-tip', type=float, required=True, help='Tip work function')
    check_parser.add_argument('--phi-samp', type=float, required=True, help='Sample work function')
    check_parser.add_argument('--n-E', type=int, required=True, help='Number of energy points')
    check_parser.add_argument('--n-V', type=int, required=True, help='Number of voltage points')
    check_parser.add_argument('--n-Z', type=int, required=True, help='Number of z points')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        list_luts(args.fer_output, show_details=not args.brief)
    elif args.command == 'cleanup':
        cleanup_old_luts(args.fer_output, args.keep)
    elif args.command == 'check':
        check_compatibility(args.phi_tip, args.phi_samp, args.n_E, args.n_V, args.n_Z, args.fer_output)
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 