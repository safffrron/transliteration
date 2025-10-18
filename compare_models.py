import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict

# ================================
# LOAD EVALUATION RESULTS
# ================================

def load_all_results():
    """Load evaluation results from all 4 models."""
    
    results = {}
    
    # Define model paths
    models = {
        "LSTM": "lstm_evaluation_results",
        "Transformer": "transformer_evaluation_results",
        "NVIDIA_Nemo": "nvidia_nemo_evaluation_results",
        "NVIDIA_Small": "nvidia_small_evaluation_results"
    }
    
    for model_name, result_dir in models.items():
        try:
            # Load overall metrics
            with open(f"{result_dir}/overall_metrics.json", "r", encoding="utf-8") as f:
                overall = json.load(f)
            
            # Load detailed results
            detailed = pd.read_csv(f"{result_dir}/detailed_results.csv")
            
            # Load length analysis
            length_analysis = pd.read_csv(f"{result_dir}/length_analysis.csv")
            
            results[model_name] = {
                "overall": overall,
                "detailed": detailed,
                "length_analysis": length_analysis
            }
            print(f"✓ Loaded {model_name} results")
        except Exception as e:
            print(f"✗ Could not load {model_name}: {e}")
    
    return results

# ================================
# COMPARISON ANALYSES
# ================================

def create_overall_comparison(results):
    """Create overall metrics comparison table."""
    
    comparison_data = []
    
    for model_name, data in results.items():
        metrics = data["overall"]
        comparison_data.append({
            "Model": model_name,
            "Accuracy (%)": f"{metrics['accuracy']*100:.2f}",
            "CER": f"{metrics['average_cer']:.4f}",
            "Similarity": f"{metrics['average_similarity']:.4f}",
            "Char Accuracy": f"{metrics['average_char_accuracy']:.4f}",
            "Avg Edit Dist": f"{metrics['average_edit_distance']:.2f}",
            "Inference (ms)": f"{metrics['average_inference_time_ms']:.2f}",
            "Throughput (w/s)": f"{metrics['throughput_words_per_sec']:.2f}"
        })
    
    df = pd.DataFrame(comparison_data)
    return df

def analyze_agreement_matrix(results):
    """Calculate agreement between models."""
    
    model_names = list(results.keys())
    n = len(model_names)
    
    # Get common indices
    all_indices = set(results[model_names[0]]["detailed"]["index"])
    for model in model_names[1:]:
        all_indices &= set(results[model]["detailed"]["index"])
    
    agreement_matrix = np.zeros((n, n))
    
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            if i == j:
                agreement_matrix[i][j] = 1.0
            else:
                # Get predictions for common indices
                df1 = results[model1]["detailed"].set_index("index")
                df2 = results[model2]["detailed"].set_index("index")
                
                common_preds1 = df1.loc[list(all_indices), "prediction"]
                common_preds2 = df2.loc[list(all_indices), "prediction"]
                
                agreement = (common_preds1 == common_preds2).sum() / len(all_indices)
                agreement_matrix[i][j] = agreement
    
    agreement_df = pd.DataFrame(
        agreement_matrix, 
        index=model_names, 
        columns=model_names
    )
    
    return agreement_df

def find_disagreement_cases(results, n_examples=20):
    """Find cases where models disagree."""
    
    model_names = list(results.keys())
    
    # Get common indices
    all_indices = set(results[model_names[0]]["detailed"]["index"])
    for model in model_names[1:]:
        all_indices &= set(results[model]["detailed"]["index"])
    
    disagreements = []
    
    for idx in all_indices:
        predictions = {}
        reference = None
        source = None
        
        for model in model_names:
            row = results[model]["detailed"][results[model]["detailed"]["index"] == idx].iloc[0]
            predictions[model] = row["prediction"]
            if reference is None:
                reference = row["reference"]
                source = row["source"]
        
        # Check if all predictions are different
        unique_preds = set(predictions.values())
        
        if len(unique_preds) > 1:  # At least one disagreement
            correct_models = [m for m in model_names if predictions[m] == reference]
            disagreements.append({
                "source": source,
                "reference": reference,
                **{f"{m}_pred": predictions[m] for m in model_names},
                "correct_models": ", ".join(correct_models) if correct_models else "None",
                "num_unique": len(unique_preds)
            })
    
    # Sort by number of unique predictions (most diverse first)
    disagreements.sort(key=lambda x: x["num_unique"], reverse=True)
    
    return pd.DataFrame(disagreements[:n_examples])

def analyze_complementarity(results):
    """Analyze which models complement each other."""
    
    model_names = list(results.keys())
    
    # Get common indices
    all_indices = set(results[model_names[0]]["detailed"]["index"])
    for model in model_names[1:]:
        all_indices &= set(results[model]["detailed"]["index"])
    
    complementarity = {}
    
    for i, model1 in enumerate(model_names):
        for model2 in model_names[i+1:]:
            df1 = results[model1]["detailed"].set_index("index")
            df2 = results[model2]["detailed"].set_index("index")
            
            # Cases where model1 correct but model2 wrong
            correct1 = set(df1[df1["exact_match"] == 1].index)
            correct2 = set(df2[df2["exact_match"] == 1].index)
            
            only_m1 = len(correct1 - correct2)
            only_m2 = len(correct2 - correct1)
            both_correct = len(correct1 & correct2)
            
            # Union accuracy (at least one correct)
            union_correct = len(correct1 | correct2)
            union_accuracy = union_correct / len(all_indices)
            
            complementarity[f"{model1} + {model2}"] = {
                "only_model1_correct": only_m1,
                "only_model2_correct": only_m2,
                "both_correct": both_correct,
                "union_accuracy": union_accuracy,
                "complementarity_score": (only_m1 + only_m2) / len(all_indices)
            }
    
    return pd.DataFrame(complementarity).T

def analyze_by_difficulty(results):
    """Analyze performance on easy vs hard examples."""
    
    model_names = list(results.keys())
    
    # Get common indices
    all_indices = set(results[model_names[0]]["detailed"]["index"])
    for model in model_names[1:]:
        all_indices &= set(results[model]["detailed"]["index"])
    
    # Calculate difficulty score (average error across models)
    difficulty_scores = {}
    
    for idx in all_indices:
        errors = []
        for model in model_names:
            row = results[model]["detailed"][results[model]["detailed"]["index"] == idx].iloc[0]
            errors.append(1 - row["exact_match"])
        difficulty_scores[idx] = np.mean(errors)
    
    # Categorize by difficulty
    difficulty_bins = {
        "Easy (0-25% error)": [],
        "Medium (25-50% error)": [],
        "Hard (50-75% error)": [],
        "Very Hard (75-100% error)": []
    }
    
    for idx, score in difficulty_scores.items():
        if score <= 0.25:
            difficulty_bins["Easy (0-25% error)"].append(idx)
        elif score <= 0.5:
            difficulty_bins["Medium (25-50% error)"].append(idx)
        elif score <= 0.75:
            difficulty_bins["Hard (50-75% error)"].append(idx)
        else:
            difficulty_bins["Very Hard (75-100% error)"].append(idx)
    
    # Calculate accuracy for each model on each difficulty level
    analysis = []
    
    for difficulty, indices in difficulty_bins.items():
        if not indices:
            continue
        
        for model in model_names:
            df = results[model]["detailed"]
            subset = df[df["index"].isin(indices)]
            
            analysis.append({
                "Difficulty": difficulty,
                "Model": model,
                "Count": len(indices),
                "Accuracy": subset["exact_match"].mean()
            })
    
    return pd.DataFrame(analysis)

# ================================
# VISUALIZATION FUNCTIONS
# ================================

def plot_overall_comparison(results, output_dir):
    """Plot overall metrics comparison."""
    
    metrics_to_plot = ["accuracy", "average_similarity", "average_char_accuracy"]
    metric_names = ["Accuracy", "Similarity", "Character Accuracy"]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (metric, name) in enumerate(zip(metrics_to_plot, metric_names)):
        model_names = []
        values = []
        
        for model, data in results.items():
            model_names.append(model)
            values.append(data["overall"][metric])
        
        axes[idx].bar(model_names, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
        axes[idx].set_ylabel(name)
        axes[idx].set_title(f"{name} Comparison")
        axes[idx].set_ylim([0, 1])
        axes[idx].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for i, v in enumerate(values):
            axes[idx].text(i, v + 0.02, f"{v:.3f}", ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/overall_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved overall comparison plot")
    plt.close()

def plot_length_analysis(results, output_dir):
    """Plot performance by length for all models."""
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for model_name, data in results.items():
        length_df = data["length_analysis"]
        
        # Extract range midpoints for x-axis
        x_labels = []
        x_positions = []
        accuracies = []
        
        for _, row in length_df.iterrows():
            range_str = row["length_range"]
            min_len, max_len = map(int, range_str.split("-"))
            midpoint = (min_len + max_len) / 2
            
            x_labels.append(range_str)
            x_positions.append(midpoint)
            accuracies.append(row["accuracy"])
        
        ax.plot(x_positions, accuracies, marker='o', label=model_name, linewidth=2)
    
    ax.set_xlabel("Word Length (characters)", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy by Word Length - All Models", fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/length_analysis_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved length analysis plot")
    plt.close()

def plot_agreement_heatmap(agreement_df, output_dir):
    """Plot agreement matrix as heatmap."""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        agreement_df, 
        annot=True, 
        fmt='.3f', 
        cmap='YlOrRd',
        vmin=0, 
        vmax=1,
        square=True,
        cbar_kws={'label': 'Agreement Rate'},
        ax=ax
    )
    
    ax.set_title("Model Agreement Matrix", fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/agreement_matrix.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved agreement matrix plot")
    plt.close()

def plot_inference_time(results, output_dir):
    """Plot inference time comparison."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    model_names = []
    times = []
    throughputs = []
    
    for model, data in results.items():
        model_names.append(model)
        times.append(data["overall"]["average_inference_time_ms"])
        throughputs.append(data["overall"]["throughput_words_per_sec"])
    
    x = np.arange(len(model_names))
    width = 0.35
    
    ax2 = ax.twinx()
    
    bars1 = ax.bar(x - width/2, times, width, label='Avg Time (ms)', color='steelblue')
    bars2 = ax2.bar(x + width/2, throughputs, width, label='Throughput (w/s)', color='coral')
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Average Inference Time (ms)', fontsize=12, color='steelblue')
    ax2.set_ylabel('Throughput (words/sec)', fontsize=12, color='coral')
    ax.set_title('Inference Speed Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    
    ax.tick_params(axis='y', labelcolor='steelblue')
    ax2.tick_params(axis='y', labelcolor='coral')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    fig.tight_layout()
    plt.savefig(f"{output_dir}/inference_time_comparison.png", dpi=300, bbox_inches='tight')
    print(f"✓ Saved inference time plot")
    plt.close()

# ================================
# MAIN EXECUTION
# ================================

def main():
    print("="*80)
    print("COMPREHENSIVE MODEL COMPARISON ANALYSIS")
    print("="*80)
    print()
    
    # Create output directory
    output_dir = "comparison_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all results
    print("Loading evaluation results...")
    results = load_all_results()
    
    if len(results) < 2:
        print("Error: Need at least 2 models to compare!")
        return
    
    print(f"\n✓ Successfully loaded {len(results)} models\n")
    
    # 1. Overall comparison
    print("Creating overall comparison...")
    overall_comparison = create_overall_comparison(results)
    overall_comparison.to_csv(f"{output_dir}/overall_comparison.csv", index=False)
    print(overall_comparison.to_string(index=False))
    print()
    
    # 2. Agreement matrix
    print("Calculating agreement matrix...")
    agreement_df = analyze_agreement_matrix(results)
    agreement_df.to_csv(f"{output_dir}/agreement_matrix.csv")
    print(agreement_df)
    print()
    
    # 3. Disagreement cases
    print("Finding disagreement cases...")
    disagreements = find_disagreement_cases(results, n_examples=20)
    disagreements.to_csv(f"{output_dir}/disagreement_cases.csv", index=False, encoding="utf-8")
    print(f"Found {len(disagreements)} interesting disagreement cases")
    print()
    
    # 4. Complementarity analysis
    print("Analyzing model complementarity...")
    complementarity = analyze_complementarity(results)
    complementarity.to_csv(f"{output_dir}/complementarity_analysis.csv")
    print(complementarity)
    print()
    
    # 5. Difficulty analysis
    print("Analyzing performance by difficulty...")
    difficulty_analysis = analyze_by_difficulty(results)
    difficulty_analysis.to_csv(f"{output_dir}/difficulty_analysis.csv", index=False)
    pivot = difficulty_analysis.pivot(index="Difficulty", columns="Model", values="Accuracy")
    print(pivot)
    print()
    
    # Generate visualizations
    print("Generating visualizations...")
    plot_overall_comparison(results, output_dir)
    plot_length_analysis(results, output_dir)
    plot_agreement_heatmap(agreement_df, output_dir)
    plot_inference_time(results, output_dir)
    
    # Save comprehensive report
    print("\nGenerating comprehensive report...")
    with open(f"{output_dir}/comprehensive_report.txt", "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("COMPREHENSIVE MODEL COMPARISON REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. OVERALL PERFORMANCE COMPARISON\n")
        f.write("-"*80 + "\n")
        f.write(overall_comparison.to_string(index=False))
        
        f.write("\n\n2. MODEL AGREEMENT MATRIX\n")
        f.write("-"*80 + "\n")
        f.write(agreement_df.to_string())
        
        f.write("\n\n3. COMPLEMENTARITY ANALYSIS\n")
        f.write("-"*80 + "\n")
        f.write(complementarity.to_string())
        
        f.write("\n\n4. PERFORMANCE BY DIFFICULTY\n")
        f.write("-"*80 + "\n")
        f.write(pivot.to_string())
        
        f.write("\n\n5. TOP DISAGREEMENT CASES\n")
        f.write("-"*80 + "\n")
        for idx, row in disagreements.head(10).iterrows():
            f.write(f"\nSource: {row['source']}\n")
            f.write(f"Reference: {row['reference']}\n")
            for model in results.keys():
                col_name = f"{model}_pred"
                if col_name in row:
                    f.write(f"{model:15s}: {row[col_name]}\n")
            f.write(f"Correct models: {row['correct_models']}\n")
            f.write("-"*80 + "\n")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nAll results saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - overall_comparison.csv")
    print("  - agreement_matrix.csv")
    print("  - disagreement_cases.csv")
    print("  - complementarity_analysis.csv")
    print("  - difficulty_analysis.csv")
    print("  - comprehensive_report.txt")
    print("  - Various PNG plots")
    print("="*80)

if __name__ == "__main__":
    main()