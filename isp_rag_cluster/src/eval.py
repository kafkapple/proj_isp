
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import json
import pandas as pd

def save_prediction_stats(invalid_stats, output_dir, model_name, labels):
    """Save prediction statistics"""
    try:
        # Save statistics file
        stats_file = output_dir / f'prediction_stats_{model_name}.json'
        with open(stats_file, 'w', encoding='ascii') as f:
            json.dump(invalid_stats, f, indent=2, ensure_ascii=True)
    except Exception as e:
        print(f"Error saving prediction stats: {e}")

def save_metrics(df, cfg, model_name, output_dir):
    """Save prediction statistics and metrics"""
    print("\nData preprocessing statistics (before):")
    print(f"Total samples: {len(df)}")
    print(f"Missing values:\n{df.isna().sum()}")
    print(f"Duplicates: {df.duplicated().sum()}")
    
    # Convert emotion labels to lowercase
    df['emotion'] = df['emotion'].str.lower()
    df[f'predicted_emotion_{model_name}'] = df[f'predicted_emotion_{model_name}'].str.lower()
    
    # Remove missing values and duplicates
    df = df.dropna(subset=['emotion', f'predicted_emotion_{model_name}'])
    df = df.drop_duplicates()
    
    # Use label list defined in config (convert ListConfig to regular list)
    labels = list(cfg.data.datasets[cfg.data.name].labels)
    default_emotion = str(cfg.data.default_emotion)
    
    # Get unique predicted labels
    predicted_labels = df[f'predicted_emotion_{model_name}'].unique()
    
    # Find invalid predictions (not in defined labels)
    invalid_predictions = df[~df[f'predicted_emotion_{model_name}'].isin(labels)]
    
    # Create invalid predictions statistics
    invalid_stats = {
        "total_samples": len(df),
        "invalid_count": len(invalid_predictions),
        "invalid_percentage": (len(invalid_predictions) / len(df)) * 100 if len(df) > 0 else 0,
        "invalid_labels": [label for label in predicted_labels if label not in labels],
        "samples": []
    }
    
    # Add detailed information for each invalid prediction
    for _, row in invalid_predictions.iterrows():
        invalid_stats["samples"].append({
            "text": row['text'],
            "true_emotion": row['emotion'],
            "predicted_emotion": row[f'predicted_emotion_{model_name}'],
            "confidence_score": row.get(f'confidence_score_{model_name}', None),
            "explanation": row.get(f'explanation_{model_name}', None)
        })
    
    # Save invalid predictions statistics
    invalid_file = output_dir / f'invalid_predictions_{model_name}.json'
    with open(invalid_file, 'w', encoding='utf-8') as f:
        json.dump(invalid_stats, f, indent=2, ensure_ascii=False)
    
    # Select only valid samples (predictions in defined labels)
    valid_samples = df[df[f'predicted_emotion_{model_name}'].isin(labels)]
    
    print("\nPrediction validation statistics:")
    print(f"Total samples: {len(df)}")
    print(f"Valid predictions: {len(valid_samples)}")
    print(f"Invalid predictions: {len(invalid_predictions)}")
    print(f"Invalid labels found: {invalid_stats['invalid_labels']}")
    
    # Extract actual and predicted labels for valid samples only
    y_true = valid_samples['emotion']
    y_pred = valid_samples[f'predicted_emotion_{model_name}']
    
    print(f"\nLabel information:")
    print(f"Defined labels: {labels}")
    print(f"Actual label unique values: {y_true.unique()}")
    print(f"Predicted label unique values (valid only): {y_pred.unique()}")
    
    # Generate classification report for valid samples
    report_dict = classification_report(y_true, y_pred, 
                                      labels=labels,
                                      output_dict=True, 
                                      zero_division=0)
    report_str = classification_report(y_true, y_pred, 
                                     labels=labels,
                                     zero_division=0)
    
    # Save classification report as text file
    with open(output_dir / f'classification_report_{model_name}.txt', "w", encoding='utf-8') as file:
        file.write(f"Total samples: {len(df)}\n")
        file.write(f"Valid samples: {len(valid_samples)}\n")
        file.write(f"Invalid predictions: {len(invalid_predictions)}\n")
        file.write(f"Invalid labels found: {', '.join(invalid_stats['invalid_labels'])}\n\n")
        file.write("=== Classification Report (Valid Samples Only) ===\n")
        file.write(report_str)
    
    # Save classification report as CSV with transpose
    report_df = pd.DataFrame(report_dict).round(4)
    report_df = report_df.drop(['accuracy'], errors='ignore')  # accuracy 행이 있다면 제거
    report_df = report_df.transpose()  # 행과 열을 전치
    report_df.index.name = 'class'  # 인덱스 이름을 'metric'으로 변경
    report_df.to_csv(output_dir / f'classification_report_{model_name}.csv')
    
    # Save metrics result
    metrics_result = {
        "overall": {
            "accuracy": float(report_dict['accuracy']),
            "macro_avg": report_dict.get('macro avg', {}),
            "weighted_avg": report_dict.get('weighted avg', {})
        },
        "per_class": {
            label: report_dict[label] for label in report_dict.keys()
            if label not in ['accuracy', 'macro avg', 'weighted avg']
        },
        "invalid_predictions_info": {
            "total": len(invalid_predictions),
            "percentage": (len(invalid_predictions) / len(df)) * 100 if len(df) > 0 else 0,
            "invalid_labels": invalid_stats['invalid_labels']
        }
    }
    
    # Calculate confusion matrix for valid samples only
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_normalized = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    
    # Plot confusion matrix in two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    # Original confusion matrix
    sns.heatmap(cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=labels,
                yticklabels=labels,
                ax=ax1)
    ax1.set_title(f'Confusion Matrix - {model_name}\n(Valid samples only: {len(valid_samples)})')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # Normalized confusion matrix
    sns.heatmap(cm_normalized, 
                annot=True, 
                fmt='.2f', 
                cmap='Blues',
                xticklabels=labels,
                yticklabels=labels,
                ax=ax2)
    ax2.set_title(f'Normalized Confusion Matrix - {model_name}\n(Valid samples only: {len(valid_samples)})')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_dir / f'confusion_matrix_{model_name}.png', 
                bbox_inches='tight',
                dpi=300)
    plt.close()
    
    return metrics_result, cm