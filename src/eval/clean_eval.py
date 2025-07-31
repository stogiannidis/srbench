#!/usr/bin/env python3
"""
Simplified script to clean CSV response columns using MultipleChoiceNormalizer.
Processes all CSV files in a directory and saves cleaned responses in a new "cleaned_response" column.
"""

import argparse
import pandas as pd
import os
import sys
import logging
import glob
from typing import List

# Add the parent directory to the path to import mcn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcn import MultipleChoiceNormalizer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Clean CSV response columns in a directory")
    parser.add_argument("input_directory", help="Directory containing CSV files")
    parser.add_argument("--response-column", default="raw_response", help="Response column name (default: response)")
    parser.add_argument("--choices", nargs="+", default=["A", "B", "C", "D", "E"], help="Valid choices")
    parser.add_argument("--no-few-shot", action="store_true", help="Disable few-shot examples")
    parser.add_argument("--max-samples", type=int, help="Limit samples for testing")
    return parser.parse_args()


def find_csv_files(directory: str) -> List[str]:
    """Find all CSV files in directory."""
    csv_files = glob.glob(os.path.join(directory, "*.csv"))
    logger.info(f"Found {len(csv_files)} CSV files")
    return sorted(csv_files)


def process_csv_file(filepath: str, normalizer: MultipleChoiceNormalizer, args) -> bool:
    """Process a single CSV file."""
    try:
        logger.info(f"Processing: {os.path.basename(filepath)}")
        
        # Load CSV
        df = pd.read_csv(filepath)
        if args.max_samples:
            df = df.head(args.max_samples)
        
        # Check response column exists
        if args.response_column not in df.columns:
            logger.warning(f"Column '{args.response_column}' not found in {filepath}")
            return False
        
        # Get responses
        responses = df[args.response_column].fillna("").astype(str).tolist()
        if not responses:
            logger.warning(f"No responses found in {filepath}")
            return False
        
        # Clean responses
        logger.info(f"Cleaning {len(responses)} responses")
        cleaned_responses = normalizer.normalize_to_multiple_choice(
            responses,
            choices=args.choices,
            use_few_shot=not args.no_few_shot
        )
        
        # Save to new column instead of overwriting
        df["cleaned_response"] = cleaned_responses
        df.to_csv(filepath, index=False)
        
        # Show changes
        changes = sum(1 for orig, clean in zip(responses, cleaned_responses) 
                     if str(orig).strip() != str(clean).strip())
        logger.info(f"✓ Changed {changes}/{len(responses)} responses ({changes/len(responses)*100:.1f}%)")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to process {filepath}: {e}")
        return False


def main():
    """Main function."""
    args = parse_arguments()
    
    # Validate directory
    if not os.path.isdir(args.input_directory):
        logger.error(f"Directory not found: {args.input_directory}")
        sys.exit(1)
    
    # Find CSV files
    csv_files = find_csv_files(args.input_directory)
    if not csv_files:
        logger.error("No CSV files found")
        sys.exit(1)
    
    # Initialize normalizer
    try:
        logger.info("Initializing normalizer...")
        normalizer = MultipleChoiceNormalizer()
    except Exception as e:
        logger.error(f"Failed to initialize normalizer: {e}")
        sys.exit(1)
    
    # Process files
    successful = 0
    for i, csv_file in enumerate(csv_files, 1):
        logger.info(f"[{i}/{len(csv_files)}] {os.path.basename(csv_file)}")
        if process_csv_file(csv_file, normalizer, args):
            successful += 1
    
    # Summary
    logger.info(f"Complete: {successful}/{len(csv_files)} files processed successfully")


if __name__ == "__main__":
    main()