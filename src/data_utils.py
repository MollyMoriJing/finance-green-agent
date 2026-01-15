"""
Data validation utilities for Finance Green Agent.

Validates dataset structure, API configuration, and data integrity.
"""

import json
import os
from pathlib import Path
from typing import Any


class DatasetReport:
    """Report of dataset validation results."""
    
    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.stats: dict[str, Any] = {}
    
    @property
    def is_valid(self) -> bool:
        """Check if dataset passed validation."""
        return len(self.errors) == 0
    
    def print_report(self) -> None:
        """Print validation report to console."""
        print("\n" + "="*60)
        print("📊 Dataset Validation Report")
        print("="*60)
        
        if self.stats:
            print("\n✓ Statistics:")
            for key, value in self.stats.items():
                print(f"  {key}: {value}")
        
        if self.warnings:
            print(f"\n⚠️  Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"  - {warning}")
        
        if self.errors:
            print(f"\n❌ Errors ({len(self.errors)}):")
            for error in self.errors:
                print(f"  - {error}")
        else:
            print("\n✅ No errors found!")
        
        print("="*60 + "\n")


def check_api_key() -> bool:
    """
    Verify that OPENROUTER_API_KEY is configured.
    
    Returns:
        True if API key is set, False otherwise
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    
    if not api_key:
        print("❌ OPENROUTER_API_KEY environment variable not set!")
        print("   Please set it in your .env file or environment.")
        return False
    
    if api_key == "your-openrouter-api-key-here":
        print("⚠️  OPENROUTER_API_KEY still has default placeholder value!")
        print("   Please update it with your actual API key.")
        return False
    
    print(f"✓ API key configured (length: {len(api_key)} chars)")
    return True


def validate_dataset(base_path: str | Path = "data") -> DatasetReport:
    """
    Validate the SEC 10-K dataset structure and integrity.
    
    Args:
        base_path: Path to data directory
        
    Returns:
        DatasetReport with validation results
    """
    report = DatasetReport()
    base_path = Path(base_path)
    
    # Check if data directory exists
    if not base_path.exists():
        report.errors.append(f"Data directory not found: {base_path}")
        return report
    
    if not base_path.is_dir():
        report.errors.append(f"Data path is not a directory: {base_path}")
        return report
    
    # Expected years
    expected_years = ["2015", "2016", "2017", "2018", "2019", "2020"]
    total_files = 0
    files_by_year = {}
    missing_sections = []
    
    # Check each year
    for year in expected_years:
        year_path = base_path / year
        
        if not year_path.exists():
            report.warnings.append(f"Year directory missing: {year}")
            continue
        
        # Count JSON files
        json_files = list(year_path.glob("*.json"))
        file_count = len(json_files)
        files_by_year[year] = file_count
        total_files += file_count
        
        if file_count == 0:
            report.warnings.append(f"No JSON files in {year} directory")
        elif file_count < 50:
            report.warnings.append(f"Only {file_count} files in {year} (expected ~150)")
        
        # Sample check: validate structure of first file in each year
        if json_files:
            sample_file = json_files[0]
            try:
                with open(sample_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Check for required sections
                required_sections = ["section_1", "section_1A", "section_7"]
                for section in required_sections:
                    if section not in data or not data[section]:
                        missing_sections.append(f"{sample_file.name} missing {section}")
                
            except json.JSONDecodeError:
                report.errors.append(f"Invalid JSON in sample file: {sample_file}")
            except Exception as e:
                report.warnings.append(f"Error reading sample file {sample_file}: {e}")
    
    # Update statistics
    report.stats = {
        "Total files": total_files,
        "Years covered": len([y for y in expected_years if (base_path / y).exists()]),
        "Files by year": files_by_year
    }
    
    # Check minimum dataset size
    if total_files < 100:
        report.errors.append(f"Dataset too small: only {total_files} files (expected ~900)")
    elif total_files < 500:
        report.warnings.append(f"Dataset smaller than expected: {total_files} files")
    
    # Report missing sections
    if missing_sections:
        report.warnings.append(f"Sample files with missing sections: {len(missing_sections)}")
        for missing in missing_sections[:5]:  # Show first 5
            report.warnings.append(f"  - {missing}")
    
    return report


def validate_environment() -> bool:
    """
    Validate complete environment setup.
    
    Returns:
        True if environment is ready, False otherwise
    """
    print("\n🔍 Validating environment...\n")
    
    all_valid = True
    
    # Check API key
    if not check_api_key():
        all_valid = False
    
    # Check dataset
    report = validate_dataset()
    report.print_report()
    
    if not report.is_valid:
        all_valid = False
    
    if all_valid:
        print("✅ Environment validation passed!\n")
    else:
        print("❌ Environment validation failed - please fix errors above\n")
    
    return all_valid


if __name__ == "__main__":
    # Run validation when executed directly
    validate_environment()
