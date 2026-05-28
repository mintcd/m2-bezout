import os
import re
import csv

MODELS = ["knn", "logreg_std", "svc", "svc_maxabs"]

def parse_sde_flops(directory="out"):
    results = []
    # Matches the precision (double|single), vector width (\d+), and the count (\d+)
    flop_pattern = re.compile(r"elements_fp_(double|single)_(\d+)\s+(\d+)")

    # Print table header
    print(f"{'Model':<15} | {'Scalar Double':<15} | {'Packed Double':<15} | {'Total FLOPs':<15}")
    print("-" * 68)

    for model in MODELS:
        filename = f"{model}.txt"
        filepath = os.path.join(directory, filename)
        
        if not os.path.exists(filepath):
            print(f"File '{filename}' not found in directory '{directory}'")
            continue

        # Initialize counters for this model
        stats = {
            "Model": model,
            "Scalar_Double": 0,
            "Packed_Double": 0,
            "Scalar_Single": 0,
            "Packed_Single": 0,
            "Total_FLOPs": 0
        }

        with open(filepath, 'r') as f:
            for line in f:
                match = flop_pattern.search(line)
                if match:
                    precision = match.group(1)
                    width = int(match.group(2))
                    count = int(match.group(3))
                    flops = width * count
                    
                    stats["Total_FLOPs"] += flops
                    
                    # Categorize the operations
                    if precision == "double":
                        if width == 1:
                            stats["Scalar_Double"] += flops
                        else:
                            stats["Packed_Double"] += flops
                    elif precision == "single":
                        if width == 1:
                            stats["Scalar_Single"] += flops
                        else:
                            stats["Packed_Single"] += flops
        
        # Formatting for terminal output
        sd_fmt = f"{stats['Scalar_Double']:,}"
        pd_fmt = f"{stats['Packed_Double']:,}"
        total_fmt = f"{stats['Total_FLOPs']:,}"
        
        print(f"{model:<15} | {sd_fmt:<15} | {pd_fmt:<15} | {total_fmt:<15}")
        results.append(stats)

    # Save to CSV
    with open('model_complexity_results.csv', 'w', newline='') as csvfile:
        fieldnames = [
            'Model', 
            'Total_FLOPs', 
            'Scalar_Double', 
            'Packed_Double', 
            'Scalar_Single', 
            'Packed_Single'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nResults saved to model_complexity_results.csv")

if __name__ == "__main__":
    parse_sde_flops()