import pandas as pd
import evaluate

INPUT_PATH = "outputs/drafts/baseline_llama.csv"

def main():
    # Load the generated summaries and references from the CSV file
    df = pd.read_csv(INPUT_PATH)
    
    # Initialize the ROUGE metric
    rouge = evaluate.load("rouge")
    
    # Compute ROUGE scores
    results = rouge.compute(predictions=df["summary"].tolist(), references=df["reference"].tolist())
    
    # Print the results
    # print("ROUGE Scores:")
    # for key, value in results.items():
    #     print(f"{key}: {value:.4f}")
    
    print(results)

if __name__ == "__main__":
    main()
