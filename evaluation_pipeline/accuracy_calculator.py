#!/usr/bin/env python3
"""
Accuracy computation system for triple extraction evaluation.
Calculates various metrics based on LLM evaluation results.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import statistics

class AccuracyCalculator:
    def __init__(self, results_dir: str = "evaluation_outputs/evaluations"):
        self.results_dir = Path(results_dir)
        
    def load_evaluation_results(self, author_name: str) -> List[Dict]:
        """Load evaluation results for a specific author."""
        eval_file = self.results_dir / f"{author_name}_evaluation.json"
        
        if not eval_file.exists():
            print(f"Warning: No evaluation results found for {author_name}")
            return []
        
        with open(eval_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def calculate_author_metrics(self, author_name: str) -> Dict:
        """Calculate accuracy metrics for a single author."""
        results = self.load_evaluation_results(author_name)
        
        if not results:
            return {
                'author': author_name,
                'total_triples': 0,
                'extractable_triples': 0,
                'extraction_rate': 0.0,
                'average_confidence': 0.0,
                'confidence_std': 0.0,
                'high_confidence_triples': 0,
                'medium_confidence_triples': 0,
                'low_confidence_triples': 0,
                'error_rate': 0.0
            }
        
        total_triples = len(results)
        extractable_triples = sum(1 for r in results if r.get('extractable', False))
        extraction_rate = extractable_triples / total_triples if total_triples > 0 else 0.0
        
        confidences = [r.get('confidence', 0.0) for r in results if r.get('extractable', False)]
        average_confidence = statistics.mean(confidences) if confidences else 0.0
        confidence_std = statistics.stdev(confidences) if len(confidences) > 1 else 0.0
        
        # Confidence categories
        high_confidence = sum(1 for c in confidences if c >= 0.8)
        medium_confidence = sum(1 for c in confidences if 0.5 <= c < 0.8)
        low_confidence = sum(1 for c in confidences if c < 0.5)
        
        # Error rate
        error_count = sum(1 for r in results if r.get('error') is not None)
        error_rate = error_count / total_triples if total_triples > 0 else 0.0
        
        return {
            'author': author_name,
            'total_triples': total_triples,
            'extractable_triples': extractable_triples,
            'extraction_rate': extraction_rate,
            'average_confidence': average_confidence,
            'confidence_std': confidence_std,
            'high_confidence_triples': high_confidence,
            'medium_confidence_triples': medium_confidence,
            'low_confidence_triples': low_confidence,
            'error_rate': error_rate
        }
    
    def calculate_overall_metrics(self, author_metrics: List[Dict]) -> Dict:
        """Calculate overall metrics across all authors."""
        if not author_metrics:
            return {}
        
        total_triples = sum(m['total_triples'] for m in author_metrics)
        total_extractable = sum(m['extractable_triples'] for m in author_metrics)
        overall_extraction_rate = total_extractable / total_triples if total_triples > 0 else 0.0
        
        # Weighted average confidence
        weighted_confidence = 0.0
        total_weight = 0
        for m in author_metrics:
            if m['extractable_triples'] > 0:
                weighted_confidence += m['average_confidence'] * m['extractable_triples']
                total_weight += m['extractable_triples']
        
        overall_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.0
        
        # Error rate
        total_errors = sum(m['total_triples'] * m['error_rate'] for m in author_metrics)
        overall_error_rate = total_errors / total_triples if total_triples > 0 else 0.0
        
        # Author-level statistics
        extraction_rates = [m['extraction_rate'] for m in author_metrics if m['total_triples'] > 0]
        avg_author_extraction_rate = statistics.mean(extraction_rates) if extraction_rates else 0.0
        std_author_extraction_rate = statistics.stdev(extraction_rates) if len(extraction_rates) > 1 else 0.0
        
        return {
            'total_authors': len(author_metrics),
            'total_triples': total_triples,
            'total_extractable_triples': total_extractable,
            'overall_extraction_rate': overall_extraction_rate,
            'overall_confidence': overall_confidence,
            'overall_error_rate': overall_error_rate,
            'avg_author_extraction_rate': avg_author_extraction_rate,
            'std_author_extraction_rate': std_author_extraction_rate,
            'best_author': max(author_metrics, key=lambda x: x['extraction_rate'])['author'] if author_metrics else None,
            'worst_author': min(author_metrics, key=lambda x: x['extraction_rate'])['author'] if author_metrics else None
        }
    
    def generate_detailed_report(self, author_metrics: List[Dict], overall_metrics: Dict) -> str:
        """Generate a detailed accuracy report."""
        report = []
        report.append("=" * 80)
        report.append("RDF TRIPLE EXTRACTION ACCURACY REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall summary
        report.append("OVERALL SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Authors Processed: {overall_metrics.get('total_authors', 0)}")
        report.append(f"Total Triples Generated: {overall_metrics.get('total_triples', 0)}")
        report.append(f"Total Extractable Triples: {overall_metrics.get('total_extractable_triples', 0)}")
        report.append(f"Overall Extraction Rate: {overall_metrics.get('overall_extraction_rate', 0):.3f} ({overall_metrics.get('overall_extraction_rate', 0)*100:.1f}%)")
        report.append(f"Overall Confidence: {overall_metrics.get('overall_confidence', 0):.3f}")
        report.append(f"Overall Error Rate: {overall_metrics.get('overall_error_rate', 0):.3f} ({overall_metrics.get('overall_error_rate', 0)*100:.1f}%)")
        report.append("")
        
        # Author-level statistics
        report.append("AUTHOR-LEVEL STATISTICS")
        report.append("-" * 40)
        report.append(f"Average Author Extraction Rate: {overall_metrics.get('avg_author_extraction_rate', 0):.3f}")
        report.append(f"Standard Deviation: {overall_metrics.get('std_author_extraction_rate', 0):.3f}")
        report.append(f"Best Performing Author: {overall_metrics.get('best_author', 'N/A')}")
        report.append(f"Worst Performing Author: {overall_metrics.get('worst_author', 'N/A')}")
        report.append("")
        
        # Individual author results
        report.append("INDIVIDUAL AUTHOR RESULTS")
        report.append("-" * 40)
        report.append(f"{'Author':<25} {'Triples':<8} {'Extractable':<12} {'Rate':<8} {'Confidence':<12} {'Errors':<8}")
        report.append("-" * 80)
        
        for metrics in sorted(author_metrics, key=lambda x: x['extraction_rate'], reverse=True):
            report.append(
                f"{metrics['author']:<25} "
                f"{metrics['total_triples']:<8} "
                f"{metrics['extractable_triples']:<12} "
                f"{metrics['extraction_rate']:<8.3f} "
                f"{metrics['average_confidence']:<12.3f} "
                f"{metrics['error_rate']:<8.3f}"
            )
        
        report.append("")
        
        # Confidence distribution
        report.append("CONFIDENCE DISTRIBUTION")
        report.append("-" * 40)
        high_conf = sum(m['high_confidence_triples'] for m in author_metrics)
        med_conf = sum(m['medium_confidence_triples'] for m in author_metrics)
        low_conf = sum(m['low_confidence_triples'] for m in author_metrics)
        total_conf = high_conf + med_conf + low_conf
        
        if total_conf > 0:
            report.append(f"High Confidence (≥0.8): {high_conf} ({high_conf/total_conf*100:.1f}%)")
            report.append(f"Medium Confidence (0.5-0.8): {med_conf} ({med_conf/total_conf*100:.1f}%)")
            report.append(f"Low Confidence (<0.5): {low_conf} ({low_conf/total_conf*100:.1f}%)")
        
        report.append("")
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_metrics(self, author_metrics: List[Dict], overall_metrics: Dict, output_file: str):
        """Save metrics to JSON file."""
        output_data = {
            'author_metrics': author_metrics,
            'overall_metrics': overall_metrics,
            'generated_at': str(Path().cwd())
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    def process_all_authors(self) -> Tuple[List[Dict], Dict]:
        """Process all authors and calculate metrics."""
        # Find all evaluation files
        eval_files = list(self.results_dir.glob("*_evaluation.json"))
        author_names = [f.stem.replace("_evaluation", "") for f in eval_files]
        
        print(f"Found evaluation results for {len(author_names)} authors")
        
        # Calculate metrics for each author
        author_metrics = []
        for author_name in author_names:
            print(f"Calculating metrics for {author_name}...")
            metrics = self.calculate_author_metrics(author_name)
            author_metrics.append(metrics)
        
        # Calculate overall metrics
        overall_metrics = self.calculate_overall_metrics(author_metrics)
        
        return author_metrics, overall_metrics

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Calculate accuracy metrics for triple extraction")
    parser.add_argument("--results-dir", default="evaluation_outputs/evaluations", help="Directory containing evaluation results")
    parser.add_argument("--output", default="accuracy_report.json", help="Output metrics file")
    parser.add_argument("--report", default="accuracy_report.txt", help="Output report file")
    
    args = parser.parse_args()
    
    calculator = AccuracyCalculator(args.results_dir)
    
    # Process all authors
    author_metrics, overall_metrics = calculator.process_all_authors()
    
    # Save metrics
    calculator.save_metrics(author_metrics, overall_metrics, args.output)
    
    # Generate and save report
    report = calculator.generate_detailed_report(author_metrics, overall_metrics)
    with open(args.report, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\nMetrics saved to: {args.output}")
    print(f"Report saved to: {args.report}")
    print("\n" + "="*50)
    print("QUICK SUMMARY")
    print("="*50)
    print(f"Authors: {overall_metrics.get('total_authors', 0)}")
    print(f"Total Triples: {overall_metrics.get('total_triples', 0)}")
    print(f"Extraction Rate: {overall_metrics.get('overall_extraction_rate', 0)*100:.1f}%")
    print(f"Average Confidence: {overall_metrics.get('overall_confidence', 0):.3f}")

if __name__ == "__main__":
    main()
