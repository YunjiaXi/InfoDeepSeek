import json
import os
from collections import defaultdict
import argparse


def calculate_accuracy_metrics(file_path, max_k=5):
    """Calculate accuracy metrics with k-level granularity"""
    metrics = {
        "response": {
            "overall": defaultdict(int),
            "attributes": defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0})),
            "domains": defaultdict(lambda: {"correct": 0, "total": 0}),
            "languages": defaultdict(lambda: {"correct": 0, "total": 0})
        },
        "off_response": {
            "overall": defaultdict(int),
            "attributes": defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0})),
            "domains": defaultdict(lambda: {"correct": 0, "total": 0}),
            "languages": defaultdict(lambda: {"correct": 0, "total": 0})
        },
        "off_right": 0,
        "off_on_both_right": 0,
    }
    IC_list = []
    # Initialize k-level metrics structure
    for k in range(1, max_k + 1):
        metrics[f"k{k}"] = {
            "overall": defaultdict(int),
            "attributes": defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0})),
            "domains": defaultdict(lambda: {"correct": 0, "total": 0}),
            "languages": defaultdict(lambda: {"correct": 0, "total": 0})
        }
    num = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)

                # Process response accuracy
                if isinstance(data.get("response_score"), bool):
                    update_metrics(
                        metrics["response"],
                        data,
                        data["response_score"]
                    )

                # Process offline response accuracy
                if isinstance(data.get("off_response_score"), bool):
                    update_metrics(
                        metrics["off_response"],
                        data,
                        data["off_response_score"]
                    )

                    if data.get("off_response_score"):
                        metrics['off_right'] += 1
                        if data.get("response_score"):
                            metrics['off_on_both_right'] += 1

                # Process answer_at_k accuracy
                answer_at_k = data.get("answer_at_k_score", {})
                max_len = len(answer_at_k)
                num += 1
                prev = False
                for k in range(1, max_len + 1):
                    k_acc = answer_at_k.get(f"k{k}", answer_at_k.get(f"k{max_len}"))
                    if isinstance(k_acc, bool):
                        update_metrics(
                            metrics[f"k{k}"],
                            data,
                            k_acc
                        )
                        prev = k_acc
                        # print(metrics[f"k{k}"]['overall']["total"])
                for k in range(max_len + 1, max_k + 1):
                    update_metrics(
                        metrics[f"k{k}"],
                        data,
                        prev
                    )
                # print(data.get('result').get("more_info").get("ranked_webpages", []))
                # print(data.get("sources"))
                is_valid = False
                for v in answer_at_k.values():
                    if v:
                        is_valid = True
                        break
                VEL = len(data.get('result').get("more_info").get("ranked_webpages")) if is_valid else max_k + 1
                IC_list.append(VEL / len(data.get("sources")))
            except Exception as e:
                print(f"Error processing line: {str(e)}")
                # exit()
                continue

    # Calculate final accuracy percentages
    results = {
        "response": calculate_dimensions(metrics["response"]),
        "off_response": calculate_dimensions(metrics["off_response"])
    }

    for k in range(1, max_k + 1):
        results[f"k{k}"] = calculate_dimensions(metrics[f"k{k}"])
    results['IC'] = sum(IC_list) / len(IC_list)
    results['EEU'] = max([results[f"k{k}"]['overall']['accuracy'] for k in range(1, max_k + 1)]) / results['response']['overall']['accuracy']
    results['DR'] = (1 - metrics['off_on_both_right'] / metrics['off_right']) if metrics['off_right'] else 1
    return results


def update_metrics(metrics, data, is_correct):
    """Update metrics for a single data point"""
    # Overall stats
    metrics["overall"]["total"] += 1
    if is_correct:
        metrics["overall"]["correct"] += 1

    # Attribute dimensions
    attributes = [
        "difficulty_GPT", "difficulty_DS", "multi_hop",
        "long_tail", "time_sensitive", "freshness",
        "mixed_truth", "false_premise"
    ]
    for attr in attributes:
        value = str(data.get(attr)).lower()
        metrics["attributes"][attr][value]["total"] += 1
        if is_correct:
            metrics["attributes"][attr][value]["correct"] += 1

    # Domain dimensions
    for domain in data.get("domain", []):
        metrics["domains"][domain]["total"] += 1
        if is_correct:
            metrics["domains"][domain]["correct"] += 1

    # Language dimensions
    for lang in data.get("advantage_language", []):
        metrics["languages"][lang]["total"] += 1
        if is_correct:
            metrics["languages"][lang]["correct"] += 1


def calculate_dimensions(metrics):
    """Calculate accuracy percentages for all dimensions"""
    return {
        "overall": calculate_accuracy(metrics["overall"]),
        "attributes": {
            attr: {
                val: calculate_accuracy(stats)
                for val, stats in attr_data.items()
            }
            for attr, attr_data in metrics["attributes"].items()
        },
        "domains": {
            domain: calculate_accuracy(stats)
            for domain, stats in metrics["domains"].items()
        },
        "languages": {
            lang: calculate_accuracy(stats)
            for lang, stats in metrics["languages"].items()
        }
    }


def calculate_accuracy(counts):
    """Calculate accuracy from correct/total counts"""
    if counts["total"] == 0:
        return {"accuracy": 0.0, "total": 0, "correct": 0}
    return {
        "accuracy": counts["correct"] / counts["total"],
        "correct": counts["correct"],
        "total": counts["total"]
    }


def print_results(results, print_attribute, print_domain, print_language, max_k=5):
    """Print formatted results for all k-levels"""
    # Print offline response accuracy
    print("\n=== Offline Response Accuracy ===")
    print_accuracy_section(results["off_response"], print_attribute, print_domain, print_language)

    # Print response accuracy
    print("\n=== Response Accuracy ===")
    print_accuracy_section(results["response"], print_attribute, print_domain, print_language)

    # Print k-level accuracies
    for k in range(1, max_k + 1):
        print(f"\n=== Accuracy@{k} ===")
        print_accuracy_section(results[f"k{k}"], print_attribute, print_domain, print_language)
    print(f"\n=== IC&EEU ===")
    print('IC:', results['IC'], '\nEEU:', results['EEU'], '\nDistracted Ratio:', results['DR'])


def print_accuracy_section(section, print_attribute, print_domain, print_language):
    """Print a single accuracy section"""
    print(f"\nOverall Accuracy: {section['overall']['accuracy']:.2%} (n={section['overall']['total']})")

    if print_attribute:
        print("\nBy Attributes:")
        for attr, values in section["attributes"].items():
            print(f"\n{attr.upper():<20}")
            for val, stats in values.items():
                print(f"{val:<8}: {stats['accuracy']:.2%} (n={stats['total']})")

    if print_domain:
        print("\nBy Domains:")
        for domain, stats in section["domains"].items():
            print(f"{domain.upper():<20}: {stats['accuracy']:.2%} (n={stats['total']})")

    if print_language:
        print("\nBy Languages:")
        for lang, stats in section["languages"].items():
            print(f"{lang.upper():<20}: {stats['accuracy']:.2%} (n={stats['total']})")


if __name__ == "__main__":
    # Example usage
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, help="file for evaluation")
    parser.add_argument("--output_file", type=str, help="output file")
    parser.add_argument("--print_attribute", default=False, action='store_true', help="whether print accuracy per attribute")
    parser.add_argument("--print_domain", default=False, action='store_true', help="whether print accuracy per domain")
    parser.add_argument("--print_language", default=False, action='store_true', help="whether print accuracy per language")
    args = parser.parse_args()

    results = calculate_accuracy_metrics(args.input_file)
    print_results(results, args.print_attribute, args.print_domain, args.print_language)

    if not args.output_file:
        idx = os.path.basename(args.input_file).rfind('.json')
        args.output_file = os.path.join(os.path.dirname(args.input_file),
                                        os.path.basename(args.input_file)[:idx] + '_acc.json')
    # Optional: Save results to JSON file
    with open(args.output_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to {args.output_file}")
